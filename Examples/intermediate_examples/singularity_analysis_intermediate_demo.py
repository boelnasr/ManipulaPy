#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Intermediate Singularity Analysis Demo - ManipulaPy

This demo showcases comprehensive singularity analysis capabilities including:
- Singularity detection and classification
- Manipulability ellipsoid visualization
- Condition number analysis
- Workspace boundary analysis
- Singularity avoidance strategies
- Real-time singularity monitoring
- Dexterity measures and optimization
- GPU-accelerated workspace sampling

Copyright (c) 2025 Mohamed Aboelnar
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import logging
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')

# ManipulaPy imports
try:
    from ManipulaPy.kinematics import SerialManipulator
    from ManipulaPy.dynamics import ManipulatorDynamics
    from ManipulaPy.singularity import Singularity
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
    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
    GPU_AVAILABLE = True
    print("✅ GPU acceleration available")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ GPU acceleration not available, using CPU only")

# Scientific computing imports
try:
    from scipy.linalg import svd, qr
    from scipy.spatial import ConvexHull
    from scipy.optimize import minimize
    from sklearn.cluster import DBSCAN
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ SciPy/sklearn not available, some features will be limited")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntermediateSingularityDemo:
    """
    Demonstrates advanced singularity analysis techniques for robotic manipulators.
    """
    
    def __init__(self, use_simple_robot=False):
        """
        Initialize the singularity analysis demo.
        
        Args:
            use_simple_robot: If True, creates a simple 3-DOF robot. 
                             If False (default), uses the built-in XArm robot.
        """
        self.use_simple_robot = use_simple_robot
        self.setup_robot()
        self.setup_singularity_analyzer()
        
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
            
            logger.info(f"✅ Loaded {num_joints}-DOF XArm robot successfully")
            logger.info(f"   Joint limits: {self.joint_limits.shape}")
            
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
        
        logger.info("✅ Simple robot setup complete")
        
    def setup_singularity_analyzer(self):
        """Initialize the singularity analysis tools."""
        logger.info("Setting up singularity analyzer...")
        
        try:
            self.singularity_analyzer = Singularity(self.robot)
            logger.info("✅ Singularity analyzer initialized")
            
            # Check GPU availability for workspace analysis
            if check_cuda_availability():
                logger.info("🚀 GPU acceleration available for workspace analysis")
            else:
                logger.info("⚙️ Using CPU-only workspace analysis")
                
        except Exception as e:
            logger.error(f"Failed to initialize singularity analyzer: {e}")
            raise
    
    def demonstrate_basic_singularity_detection(self):
        """Demonstrate basic singularity detection and analysis."""
        logger.info("\n🎯 Demonstrating Basic Singularity Detection...")
        
        num_joints = len(self.joint_limits)
        
        # Test configurations including known singularities
        test_configs = []
        
        # 1. Home position (typically not singular)
        test_configs.append(("Home", np.zeros(num_joints)))
        
        # 2. Extended configuration (potentially singular)
        extended_config = np.array([
            (self.joint_limits[i, 1] - self.joint_limits[i, 0]) * 0.8 + self.joint_limits[i, 0]
            for i in range(num_joints)
        ])
        test_configs.append(("Extended", extended_config))
        
        # 3. Folded configuration (potentially singular)
        folded_config = np.array([
            (self.joint_limits[i, 1] - self.joint_limits[i, 0]) * 0.1 + self.joint_limits[i, 0]
            for i in range(num_joints)
        ])
        test_configs.append(("Folded", folded_config))
        
        # 4. Random configurations
        np.random.seed(42)
        for i in range(3):
            random_config = np.array([
                np.random.uniform(self.joint_limits[j, 0], self.joint_limits[j, 1])
                for j in range(num_joints)
            ])
            test_configs.append((f"Random{i+1}", random_config))
        
        # Analyze each configuration
        singularity_results = []
        
        for name, config in test_configs:
            logger.info(f"  Analyzing configuration: {name}")
            
            # Compute Jacobian
            J = self.robot.jacobian(config)
            
            # Singularity analysis
            is_singular = self.singularity_analyzer.singularity_analysis(config)
            condition_number = self.singularity_analyzer.condition_number(config)
            near_singular = self.singularity_analyzer.near_singularity_detection(config)
            
            # Additional metrics
            det_J = np.linalg.det(J @ J.T)  # Manipulability measure
            U, s, Vt = svd(J)
            min_singular_value = np.min(s)
            max_singular_value = np.max(s)
            
            result = {
                'name': name,
                'config': config,
                'jacobian': J,
                'is_singular': is_singular,
                'condition_number': condition_number,
                'near_singular': near_singular,
                'manipulability': np.sqrt(det_J),
                'min_sv': min_singular_value,
                'max_sv': max_singular_value,
                'singular_values': s
            }
            
            singularity_results.append(result)
            
            # Log results
            logger.info(f"    Singular: {is_singular}")
            logger.info(f"    Near singular: {near_singular}")
            logger.info(f"    Condition number: {condition_number:.4f}")
            logger.info(f"    Manipulability: {result['manipulability']:.6f}")
            logger.info(f"    Min singular value: {min_singular_value:.6f}")
        
        # Plot comparison
        self.plot_singularity_comparison(singularity_results)
        
        # Analyze singular value patterns
        self.analyze_singular_value_patterns(singularity_results)
        
        logger.info("✅ Basic singularity detection demonstration complete")
        
        return singularity_results
    
    def plot_singularity_comparison(self, results):
        """Plot comparison of singularity metrics across configurations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Singularity Analysis Comparison', fontsize=16, fontweight='bold')
        
        names = [r['name'] for r in results]
        condition_numbers = [r['condition_number'] for r in results]
        manipulabilities = [r['manipulability'] for r in results]
        min_svs = [r['min_sv'] for r in results]
        max_svs = [r['max_sv'] for r in results]
        
        # Color code based on singularity
        colors = ['red' if r['is_singular'] else 'orange' if r['near_singular'] else 'green' 
                 for r in results]
        
        # Condition number
        ax = axes[0, 0]
        bars = ax.bar(names, condition_numbers, color=colors, alpha=0.8)
        ax.set_title('Condition Number', fontweight='bold')
        ax.set_ylabel('Condition Number')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, val in zip(bars, condition_numbers):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                   f'{val:.1f}', ha='center', va='bottom')
        
        # Manipulability
        ax = axes[0, 1]
        bars = ax.bar(names, manipulabilities, color=colors, alpha=0.8)
        ax.set_title('Manipulability Measure', fontweight='bold')
        ax.set_ylabel('Manipulability')
        ax.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, val in zip(bars, manipulabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                   f'{val:.3f}', ha='center', va='bottom')
        
        # Singular value ratio
        ax = axes[1, 0]
        sv_ratios = [max_sv/min_sv if min_sv > 1e-10 else np.inf for min_sv, max_sv in zip(min_svs, max_svs)]
        bars = ax.bar(names, sv_ratios, color=colors, alpha=0.8)
        ax.set_title('Singular Value Ratio (σ_max/σ_min)', fontweight='bold')
        ax.set_ylabel('Ratio')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Singular values spectrum
        ax = axes[1, 1]
        for i, result in enumerate(results):
            sv = result['singular_values']
            x_pos = np.arange(len(sv)) + i * 0.1
            ax.bar(x_pos, sv, width=0.1, label=result['name'], alpha=0.8)
        
        ax.set_title('Singular Values Spectrum', fontweight='bold')
        ax.set_xlabel('Singular Value Index')
        ax.set_ylabel('Value')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add legend for color coding
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.8, label='Singular'),
            Patch(facecolor='orange', alpha=0.8, label='Near Singular'),
            Patch(facecolor='green', alpha=0.8, label='Non-Singular')
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        # Get the directory of the current script
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, 'singularity_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"📊 Singularity comparison plot saved as '{save_path}'")
        plt.close()
    
    def analyze_singular_value_patterns(self, results):
        """Analyze patterns in singular values across configurations."""
        logger.info("\n📊 Singular Value Pattern Analysis:")
        
        for result in results:
            sv = result['singular_values']
            name = result['name']
            
            # Calculate metrics
            sv_ratio = sv[0] / sv[-1] if sv[-1] > 1e-10 else np.inf
            sv_gaps = np.diff(sv)
            largest_gap_idx = np.argmax(sv_gaps)
            
            logger.info(f"  {name}:")
            logger.info(f"    Singular values: {sv}")
            logger.info(f"    Max/Min ratio: {sv_ratio:.2f}")
            logger.info(f"    Largest gap: {sv_gaps[largest_gap_idx]:.6f} (between σ_{largest_gap_idx} and σ_{largest_gap_idx+1})")
            
            # Identify rank deficiency
            tolerance = 1e-6
            rank = np.sum(sv > tolerance)
            logger.info(f"    Effective rank: {rank}/{len(sv)}")
    
    def demonstrate_manipulability_ellipsoid_analysis(self):
        """Demonstrate manipulability ellipsoid visualization and analysis."""
        logger.info("\n🎯 Demonstrating Manipulability Ellipsoid Analysis...")
        
        num_joints = len(self.joint_limits)
        
        # Select representative configurations
        configs = {
            'Home': np.zeros(num_joints),
            'Mid-range': np.array([(limit[0] + limit[1]) / 2 for limit in self.joint_limits]),
            'Extended': np.array([limit[1] * 0.8 for limit in self.joint_limits]),
        }
        
        ellipsoid_data = {}
        
        for name, config in configs.items():
            logger.info(f"  Computing manipulability ellipsoid for: {name}")
            
            # Get Jacobian
            J = self.robot.jacobian(config)
            
            # Split into linear and angular parts
            J_v = J[:3, :]  # Linear velocity
            J_w = J[3:, :]  # Angular velocity
            
            # Compute manipulability ellipsoids
            ellipsoid_data[name] = {
                'config': config,
                'jacobian': J,
                'linear_jacobian': J_v,
                'angular_jacobian': J_w,
                'linear_ellipsoid': self.compute_ellipsoid_data(J_v),
                'angular_ellipsoid': self.compute_ellipsoid_data(J_w),
                'manipulability_linear': np.sqrt(np.linalg.det(J_v @ J_v.T)),
                'manipulability_angular': np.sqrt(np.linalg.det(J_w @ J_w.T)),
            }
            
            logger.info(f"    Linear manipulability: {ellipsoid_data[name]['manipulability_linear']:.6f}")
            logger.info(f"    Angular manipulability: {ellipsoid_data[name]['manipulability_angular']:.6f}")
        
        # Visualize ellipsoids
        self.plot_manipulability_ellipsoids(ellipsoid_data)
        
        # Analyze ellipsoid properties
        self.analyze_ellipsoid_properties(ellipsoid_data)
        
        logger.info("✅ Manipulability ellipsoid analysis demonstration complete")
        
        return ellipsoid_data
    
    def compute_ellipsoid_data(self, J):
        """Compute ellipsoid data from Jacobian matrix."""
        # Compute JJ^T for velocity ellipsoid
        A = J @ J.T
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Semi-axes lengths (square root of eigenvalues)
        semi_axes = np.sqrt(np.maximum(eigenvalues, 1e-10))
        
        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'semi_axes': semi_axes,
            'volume': np.prod(semi_axes) * 4/3 * np.pi if len(semi_axes) == 3 else np.prod(semi_axes) * np.pi
        }
    
    def plot_manipulability_ellipsoids(self, ellipsoid_data):
        """Plot manipulability ellipsoids for different configurations."""
        fig = plt.figure(figsize=(15, 10))
        
        # Linear velocity ellipsoids
        ax1 = fig.add_subplot(221, projection='3d')
        colors = ['blue', 'red', 'green']
        
        for i, (name, data) in enumerate(ellipsoid_data.items()):
            ellipsoid = data['linear_ellipsoid']
            if len(ellipsoid['semi_axes']) >= 3:
                self.plot_ellipsoid_3d(ax1, ellipsoid, colors[i], name, alpha=0.3)
        
        ax1.set_title('Linear Velocity Ellipsoids', fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        
        # Angular velocity ellipsoids
        ax2 = fig.add_subplot(222, projection='3d')
        
        for i, (name, data) in enumerate(ellipsoid_data.items()):
            ellipsoid = data['angular_ellipsoid']
            if len(ellipsoid['semi_axes']) >= 3:
                self.plot_ellipsoid_3d(ax2, ellipsoid, colors[i], name, alpha=0.3)
        
        ax2.set_title('Angular Velocity Ellipsoids', fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()
        
        # Manipulability comparison
        ax3 = fig.add_subplot(223)
        names = list(ellipsoid_data.keys())
        linear_manip = [data['manipulability_linear'] for data in ellipsoid_data.values()]
        angular_manip = [data['manipulability_angular'] for data in ellipsoid_data.values()]
        
        x = np.arange(len(names))
        width = 0.35
        
        ax3.bar(x - width/2, linear_manip, width, label='Linear', alpha=0.8, color='skyblue')
        ax3.bar(x + width/2, angular_manip, width, label='Angular', alpha=0.8, color='lightcoral')
        
        ax3.set_title('Manipulability Measures', fontweight='bold')
        ax3.set_xlabel('Configuration')
        ax3.set_ylabel('Manipulability')
        ax3.set_xticks(x)
        ax3.set_xticklabels(names)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Ellipsoid volume comparison
        ax4 = fig.add_subplot(224)
        linear_volumes = [data['linear_ellipsoid']['volume'] for data in ellipsoid_data.values()]
        angular_volumes = [data['angular_ellipsoid']['volume'] for data in ellipsoid_data.values()]
        
        ax4.bar(x - width/2, linear_volumes, width, label='Linear', alpha=0.8, color='skyblue')
        ax4.bar(x + width/2, angular_volumes, width, label='Angular', alpha=0.8, color='lightcoral')
        
        ax4.set_title('Ellipsoid Volumes', fontweight='bold')
        ax4.set_xlabel('Configuration')
        ax4.set_ylabel('Volume')
        ax4.set_xticks(x)
        ax4.set_xticklabels(names)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Get the directory of the current script
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, 'manipulability_ellipsoids.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"📊 Manipulability ellipsoids plot saved as '{save_path}'")
        plt.close()
    
    def plot_ellipsoid_3d(self, ax, ellipsoid_data, color, label, alpha=0.3):
        """Plot a 3D ellipsoid on given axes."""
        # Generate sphere points
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Transform sphere to ellipsoid
        semi_axes = ellipsoid_data['semi_axes']
        eigenvectors = ellipsoid_data['eigenvectors']
        
        if len(semi_axes) >= 3:
            # Scale by semi-axes
            x_ellipsoid = x_sphere * semi_axes[0]
            y_ellipsoid = y_sphere * semi_axes[1]
            z_ellipsoid = z_sphere * semi_axes[2]
            
            # Rotate by eigenvectors
            points = np.stack([x_ellipsoid.flatten(), y_ellipsoid.flatten(), z_ellipsoid.flatten()])
            rotated_points = eigenvectors @ points
            
            x_final = rotated_points[0].reshape(x_sphere.shape)
            y_final = rotated_points[1].reshape(x_sphere.shape)
            z_final = rotated_points[2].reshape(x_sphere.shape)
            
            ax.plot_surface(x_final, y_final, z_final, color=color, alpha=alpha, label=label)
    
    def analyze_ellipsoid_properties(self, ellipsoid_data):
        """Analyze properties of manipulability ellipsoids."""
        logger.info("\n📊 Ellipsoid Properties Analysis:")
        
        for name, data in ellipsoid_data.items():
            logger.info(f"  {name} Configuration:")
            
            # Linear ellipsoid analysis
            linear_ellipsoid = data['linear_ellipsoid']
            linear_axes = linear_ellipsoid['semi_axes']
            
            if len(linear_axes) >= 3:
                logger.info(f"    Linear ellipsoid semi-axes: {linear_axes}")
                logger.info(f"    Linear isotropy ratio: {np.min(linear_axes)/np.max(linear_axes):.4f}")
                logger.info(f"    Linear volume: {linear_ellipsoid['volume']:.6f}")
            
            # Angular ellipsoid analysis
            angular_ellipsoid = data['angular_ellipsoid']
            angular_axes = angular_ellipsoid['semi_axes']
            
            if len(angular_axes) >= 3:
                logger.info(f"    Angular ellipsoid semi-axes: {angular_axes}")
                logger.info(f"    Angular isotropy ratio: {np.min(angular_axes)/np.max(angular_axes):.4f}")
                logger.info(f"    Angular volume: {angular_ellipsoid['volume']:.6f}")
    
    def demonstrate_workspace_analysis(self):
        """Demonstrate workspace analysis with singularity detection."""
        logger.info("\n🎯 Demonstrating Workspace Analysis...")
        
        # Sample workspace using Monte Carlo method
        num_samples = 10000 if GPU_AVAILABLE else 5000
        logger.info(f"Sampling workspace with {num_samples} configurations...")
        
        workspace_data = self.sample_workspace(num_samples)
        
        # Analyze workspace properties
        self.analyze_workspace_properties(workspace_data)
        
        # Visualize workspace with singularities
        self.plot_workspace_analysis(workspace_data)
        
        # Find workspace boundaries
        self.find_workspace_boundaries(workspace_data)
        
        logger.info("✅ Workspace analysis demonstration complete")
        
        return workspace_data
    
    def sample_workspace(self, num_samples):
        """Sample the robot workspace and analyze singularities."""
        if GPU_AVAILABLE and CUDA_AVAILABLE:
            return self.sample_workspace_gpu(num_samples)
        else:
            return self.sample_workspace_cpu(num_samples)
    
    def sample_workspace_cpu(self, num_samples):
        """CPU-based workspace sampling."""
        logger.info("Using CPU for workspace sampling...")
        
        num_joints = len(self.joint_limits)
        
        # Generate random joint configurations
        np.random.seed(42)
        joint_configs = np.zeros((num_samples, num_joints))
        
        for i in range(num_joints):
            joint_configs[:, i] = np.random.uniform(
                self.joint_limits[i, 0], 
                self.joint_limits[i, 1], 
                num_samples
            )
        
        # Compute workspace points and singularity metrics
        workspace_points = []
        singularity_metrics = []
        
        for i in range(num_samples):
            if i % 1000 == 0:
                logger.info(f"  Processed {i}/{num_samples} configurations...")
            
            config = joint_configs[i]
            
            # Forward kinematics
            T = self.robot.forward_kinematics(config)
            ee_position = T[:3, 3]
            workspace_points.append(ee_position)
            
            # Singularity analysis
            J = self.robot.jacobian(config)
            condition_number = np.linalg.cond(J)
            manipulability = np.sqrt(np.linalg.det(J @ J.T))
            
            # Singular values
            U, s, Vt = svd(J)
            min_sv = np.min(s)
            
            singularity_metrics.append({
                'config': config,
                'condition_number': condition_number,
                'manipulability': manipulability,
                'min_singular_value': min_sv,
                'is_singular': min_sv < 1e-4,
                'near_singular': condition_number > 100
            })
        
        return {
            'joint_configs': joint_configs,
            'workspace_points': np.array(workspace_points),
            'singularity_metrics': singularity_metrics,
            'method': 'CPU'
        }
    
    def sample_workspace_gpu(self, num_samples):
        """GPU-accelerated workspace sampling."""
        logger.info("Using GPU for workspace sampling...")
        
        try:
            # Use the built-in GPU-accelerated workspace sampling
            workspace_points = self.singularity_analyzer.plot_workspace_monte_carlo(
                self.joint_limits.tolist(), num_samples
            )
            
            # For detailed analysis, we still need CPU computation
            # Sample a subset for detailed singularity analysis
            subset_size = min(1000, num_samples // 10)
            logger.info(f"Computing detailed singularity metrics for {subset_size} configurations...")
            
            return self.sample_workspace_cpu(subset_size)
            
        except Exception as e:
            logger.warning(f"GPU workspace sampling failed: {e}, falling back to CPU")
            return self.sample_workspace_cpu(num_samples)
    
    def analyze_workspace_properties(self, workspace_data):
        """Analyze properties of the sampled workspace."""
        logger.info("\n📊 Workspace Properties Analysis:")
        
        workspace_points = workspace_data['workspace_points']
        singularity_metrics = workspace_data['singularity_metrics']
        
        # Workspace volume estimation
        if SCIPY_AVAILABLE:
            try:
                hull = ConvexHull(workspace_points)
                workspace_volume = hull.volume
                workspace_area = hull.area
                logger.info(f"  Workspace convex hull volume: {workspace_volume:.6f} m³")
                logger.info(f"  Workspace convex hull surface area: {workspace_area:.6f} m²")
            except Exception as e:
                logger.warning(f"Could not compute convex hull: {e}")
        
        # Workspace extent
        min_coords = np.min(workspace_points, axis=0)
        max_coords = np.max(workspace_points, axis=0)
        workspace_extent = max_coords - min_coords
        
        logger.info(f"  Workspace extent:")
        logger.info(f"    X: [{min_coords[0]:.3f}, {max_coords[0]:.3f}] m (range: {workspace_extent[0]:.3f} m)")
        logger.info(f"    Y: [{min_coords[1]:.3f}, {max_coords[1]:.3f}] m (range: {workspace_extent[1]:.3f} m)")
        logger.info(f"    Z: [{min_coords[2]:.3f}, {max_coords[2]:.3f}] m (range: {workspace_extent[2]:.3f} m)")
        
        # Singularity statistics
        condition_numbers = [m['condition_number'] for m in singularity_metrics]
        manipulabilities = [m['manipulability'] for m in singularity_metrics]
        min_svs = [m['min_singular_value'] for m in singularity_metrics]
        
        num_singular = sum(1 for m in singularity_metrics if m['is_singular'])
        num_near_singular = sum(1 for m in singularity_metrics if m['near_singular'])
        
        logger.info(f"  Singularity statistics:")
        logger.info(f"    Singular configurations: {num_singular}/{len(singularity_metrics)} ({100*num_singular/len(singularity_metrics):.1f}%)")
        logger.info(f"    Near-singular configurations: {num_near_singular}/{len(singularity_metrics)} ({100*num_near_singular/len(singularity_metrics):.1f}%)")
        logger.info(f"    Mean condition number: {np.mean(condition_numbers):.2f}")
        logger.info(f"    Mean manipulability: {np.mean(manipulabilities):.6f}")
        logger.info(f"    Min singular value range: [{np.min(min_svs):.6f}, {np.max(min_svs):.6f}]")
    
    def plot_workspace_analysis(self, workspace_data):
        """Plot workspace analysis results."""
        workspace_points = workspace_data['workspace_points']
        singularity_metrics = workspace_data['singularity_metrics']
        
        fig = plt.figure(figsize=(20, 15))
        
        # 3D workspace with singularity coloring
        ax1 = fig.add_subplot(231, projection='3d')
        
        # Color points based on singularity
        colors = []
        for metric in singularity_metrics:
            if metric['is_singular']:
                colors.append('red')
            elif metric['near_singular']:
                colors.append('orange')
            else:
                colors.append('blue')
        
        scatter = ax1.scatter(workspace_points[:, 0], workspace_points[:, 1], workspace_points[:, 2], 
                            c=colors, s=1, alpha=0.6)
        ax1.set_title('Workspace with Singularities', fontweight='bold')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Singular'),
            Patch(facecolor='orange', label='Near Singular'),
            Patch(facecolor='blue', label='Non-Singular')
        ]
        ax1.legend(handles=legend_elements)
        
        # Condition number distribution
        ax2 = fig.add_subplot(232)
        condition_numbers = [m['condition_number'] for m in singularity_metrics]
        ax2.hist(condition_numbers, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('Condition Number Distribution', fontweight='bold')
        ax2.set_xlabel('Condition Number')
        ax2.set_ylabel('Frequency')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Manipulability distribution
        ax3 = fig.add_subplot(233)
        manipulabilities = [m['manipulability'] for m in singularity_metrics]
        ax3.hist(manipulabilities, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_title('Manipulability Distribution', fontweight='bold')
        ax3.set_xlabel('Manipulability')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Workspace projections
        ax4 = fig.add_subplot(234)
        ax4.scatter(workspace_points[:, 0], workspace_points[:, 1], c=colors, s=1, alpha=0.6)
        ax4.set_title('Workspace Projection (X-Y)', fontweight='bold')
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.set_aspect('equal', adjustable='box')
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(235)
        ax5.scatter(workspace_points[:, 0], workspace_points[:, 2], c=colors, s=1, alpha=0.6)
        ax5.set_title('Workspace Projection (X-Z)', fontweight='bold')
        ax5.set_xlabel('X (m)')
        ax5.set_ylabel('Z (m)')
        ax5.set_aspect('equal', adjustable='box')
        ax5.grid(True, alpha=0.3)
        
        # Singularity vs position correlation
        ax6 = fig.add_subplot(236)
        min_svs = [m['min_singular_value'] for m in singularity_metrics]
        distances_from_origin = np.linalg.norm(workspace_points, axis=1)
        
        ax6.scatter(distances_from_origin, min_svs, alpha=0.6, s=2)
        ax6.set_title('Min Singular Value vs Distance from Origin', fontweight='bold')
        ax6.set_xlabel('Distance from Origin (m)')
        ax6.set_ylabel('Min Singular Value')
        ax6.set_yscale('log')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Get the directory of the current script
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, 'workspace_analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"📊 Workspace analysis plot saved as '{save_path}'")
        plt.close()
    
    def find_workspace_boundaries(self, workspace_data):
        """Find and analyze workspace boundaries."""
        logger.info("\n📊 Workspace Boundary Analysis:")
        
        workspace_points = workspace_data['workspace_points']
        
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available, skipping detailed boundary analysis")
            return
        
        try:
            # Compute convex hull to find boundary
            hull = ConvexHull(workspace_points)
            
            # Get boundary points
            boundary_points = workspace_points[hull.vertices]
            
            logger.info(f"  Found {len(boundary_points)} boundary points")
            logger.info(f"  Convex hull has {len(hull.simplices)} triangular faces")
            
            # Analyze boundary singularities
            boundary_configs = []
            boundary_singularities = []
            
            for vertex_idx in hull.vertices:
                config = workspace_data['singularity_metrics'][vertex_idx]['config']
                boundary_configs.append(config)
                
                # Check if boundary point is near singular
                J = self.robot.jacobian(config)
                condition_number = np.linalg.cond(J)
                boundary_singularities.append(condition_number > 100)
            
            singular_boundary_count = sum(boundary_singularities)
            logger.info(f"  Singular boundary points: {singular_boundary_count}/{len(boundary_points)} ({100*singular_boundary_count/len(boundary_points):.1f}%)")
            
            # Plot boundary analysis
            self.plot_boundary_analysis(workspace_points, hull, boundary_singularities)
            
        except Exception as e:
            logger.error(f"Boundary analysis failed: {e}")
    
    def plot_boundary_analysis(self, workspace_points, hull, boundary_singularities):
        """Plot workspace boundary analysis."""
        fig = plt.figure(figsize=(15, 10))
        
        # 3D boundary visualization
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Plot all workspace points
        ax1.scatter(workspace_points[:, 0], workspace_points[:, 1], workspace_points[:, 2], 
                   c='lightblue', s=1, alpha=0.3, label='Workspace')
        
        # Plot boundary points with singularity coloring
        boundary_points = workspace_points[hull.vertices]
        boundary_colors = ['red' if is_sing else 'green' for is_sing in boundary_singularities]
        
        ax1.scatter(boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2], 
                   c=boundary_colors, s=20, alpha=0.8, label='Boundary')
        
        ax1.set_title('Workspace Boundary Analysis', fontweight='bold')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', alpha=0.3, label='Workspace'),
            Patch(facecolor='red', label='Singular Boundary'),
            Patch(facecolor='green', label='Non-Singular Boundary')
        ]
        ax1.legend(handles=legend_elements)
        
        # Boundary singularity distribution
        ax2 = fig.add_subplot(122)
        
        labels = ['Non-Singular', 'Singular']
        counts = [len(boundary_singularities) - sum(boundary_singularities), sum(boundary_singularities)]
        colors = ['green', 'red']
        
        wedges, texts, autotexts = ax2.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', 
                                          startangle=90, alpha=0.8)
        ax2.set_title('Boundary Point Singularity Distribution', fontweight='bold')
        
        plt.tight_layout()
        # Get the directory of the current script
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, 'boundary_analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"📊 Boundary analysis plot saved as '{save_path}'")
        plt.close()
    
    def demonstrate_singularity_avoidance(self):
        """Demonstrate singularity avoidance strategies."""
        logger.info("\n🎯 Demonstrating Singularity Avoidance Strategies...")
        
        # Define a trajectory that might encounter singularities
        num_joints = len(self.joint_limits)
        
        # Start and end configurations
        start_config = np.zeros(num_joints)
        end_config = np.array([
            (self.joint_limits[i, 1] - self.joint_limits[i, 0]) * 0.8 + self.joint_limits[i, 0]
            for i in range(num_joints)
        ])
        
        logger.info(f"Planning trajectory from {start_config} to {end_config}")
        
        # Generate different trajectory strategies
        strategies = {
            'Direct': self.plan_direct_trajectory,
            'Singularity_Aware': self.plan_singularity_aware_trajectory,
            'Damped_Least_Squares': self.plan_damped_trajectory,
        }
        
        trajectory_results = {}
        
        for strategy_name, strategy_func in strategies.items():
            logger.info(f"  Testing {strategy_name} strategy...")
            
            try:
                result = strategy_func(start_config, end_config)
                trajectory_results[strategy_name] = result
                
                # Analyze trajectory singularities
                self.analyze_trajectory_singularities(result, strategy_name)
                
            except Exception as e:
                logger.error(f"Strategy {strategy_name} failed: {e}")
        
        # Compare strategies
        self.plot_trajectory_comparison(trajectory_results)
        
        logger.info("✅ Singularity avoidance demonstration complete")
        
        return trajectory_results
    
    def plan_direct_trajectory(self, start_config, end_config, num_points=50):
        """Plan a direct linear trajectory in joint space."""
        trajectory = []
        
        for i in range(num_points):
            alpha = i / (num_points - 1)
            config = (1 - alpha) * start_config + alpha * end_config
            trajectory.append(config)
        
        return {
            'trajectory': np.array(trajectory),
            'method': 'Direct Linear Interpolation'
        }
    
    def plan_singularity_aware_trajectory(self, start_config, end_config, num_points=50):
        """Plan trajectory with singularity awareness."""
        trajectory = [start_config.copy()]
        current_config = start_config.copy()
        
        step_size = (end_config - start_config) / (num_points - 1)
        
        for i in range(1, num_points):
            # Proposed next configuration
            next_config = current_config + step_size
            
            # Check for singularities
            J = self.robot.jacobian(next_config)
            condition_number = np.linalg.cond(J)
            
            # If near singular, try to modify path
            if condition_number > 100:
                # Try different intermediate points
                best_config = next_config.copy()
                best_condition = condition_number
                
                for perturbation in [0.1, -0.1, 0.2, -0.2]:
                    perturbed_config = next_config + perturbation * np.random.randn(len(next_config)) * 0.1
                    
                    # Ensure within joint limits
                    for j in range(len(perturbed_config)):
                        perturbed_config[j] = np.clip(
                            perturbed_config[j], 
                            self.joint_limits[j, 0], 
                            self.joint_limits[j, 1]
                        )
                    
                    J_pert = self.robot.jacobian(perturbed_config)
                    cond_pert = np.linalg.cond(J_pert)
                    
                    if cond_pert < best_condition:
                        best_config = perturbed_config.copy()
                        best_condition = cond_pert
                
                next_config = best_config
            
            trajectory.append(next_config)
            current_config = next_config.copy()
        
        return {
            'trajectory': np.array(trajectory),
            'method': 'Singularity-Aware Planning'
        }
    
    def plan_damped_trajectory(self, start_config, end_config, num_points=50):
        """Plan trajectory using damped least squares approach."""
        trajectory = []
        
        # Use intermediate waypoints with damping
        for i in range(num_points):
            alpha = i / (num_points - 1)
            # Use smooth interpolation with higher-order polynomial
            alpha_smooth = 3 * alpha**2 - 2 * alpha**3  # Smooth step function
            config = (1 - alpha_smooth) * start_config + alpha_smooth * end_config
            trajectory.append(config)
        
        return {
            'trajectory': np.array(trajectory),
            'method': 'Damped Interpolation'
        }
    
    def analyze_trajectory_singularities(self, trajectory_result, strategy_name):
        """Analyze singularities along a trajectory."""
        trajectory = trajectory_result['trajectory']
        
        condition_numbers = []
        manipulabilities = []
        min_singular_values = []
        
        for config in trajectory:
            J = self.robot.jacobian(config)
            
            condition_number = np.linalg.cond(J)
            manipulability = np.sqrt(np.linalg.det(J @ J.T))
            
            U, s, Vt = svd(J)
            min_sv = np.min(s)
            
            condition_numbers.append(condition_number)
            manipulabilities.append(manipulability)
            min_singular_values.append(min_sv)
        
        # Calculate statistics
        max_condition = np.max(condition_numbers)
        mean_condition = np.mean(condition_numbers)
        min_manipulability = np.min(manipulabilities)
        mean_manipulability = np.mean(manipulabilities)
        
        num_near_singular = sum(1 for cn in condition_numbers if cn > 100)
        
        logger.info(f"    {strategy_name} trajectory analysis:")
        logger.info(f"      Max condition number: {max_condition:.2f}")
        logger.info(f"      Mean condition number: {mean_condition:.2f}")
        logger.info(f"      Min manipulability: {min_manipulability:.6f}")
        logger.info(f"      Mean manipulability: {mean_manipulability:.6f}")
        logger.info(f"      Near-singular points: {num_near_singular}/{len(trajectory)}")
        
        # Store results for plotting
        trajectory_result['condition_numbers'] = condition_numbers
        trajectory_result['manipulabilities'] = manipulabilities
        trajectory_result['min_singular_values'] = min_singular_values
    
    def plot_trajectory_comparison(self, trajectory_results):
        """Plot comparison of different trajectory strategies."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Singularity Avoidance Strategy Comparison', fontsize=16, fontweight='bold')
        
        colors = ['blue', 'red', 'green']
        
        # Condition number along trajectory
        ax = axes[0, 0]
        for i, (strategy, result) in enumerate(trajectory_results.items()):
            if 'condition_numbers' in result:
                trajectory_length = len(result['condition_numbers'])
                x = np.linspace(0, 1, trajectory_length)
                ax.plot(x, result['condition_numbers'], color=colors[i], 
                       linewidth=2, label=strategy, alpha=0.8)
        
        ax.set_title('Condition Number Along Trajectory', fontweight='bold')
        ax.set_xlabel('Trajectory Progress')
        ax.set_ylabel('Condition Number')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add singularity threshold line
        ax.axhline(y=100, color='orange', linestyle='--', alpha=0.7, label='Near-Singular Threshold')
        
        # Manipulability along trajectory
        ax = axes[0, 1]
        for i, (strategy, result) in enumerate(trajectory_results.items()):
            if 'manipulabilities' in result:
                trajectory_length = len(result['manipulabilities'])
                x = np.linspace(0, 1, trajectory_length)
                ax.plot(x, result['manipulabilities'], color=colors[i], 
                       linewidth=2, label=strategy, alpha=0.8)
        
        ax.set_title('Manipulability Along Trajectory', fontweight='bold')
        ax.set_xlabel('Trajectory Progress')
        ax.set_ylabel('Manipulability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # End-effector path comparison
        ax = axes[1, 0]
        for i, (strategy, result) in enumerate(trajectory_results.items()):
            trajectory = result['trajectory']
            ee_positions = []
            
            for config in trajectory:
                T = self.robot.forward_kinematics(config)
                ee_positions.append(T[:3, 3])
            
            ee_positions = np.array(ee_positions)
            ax.plot(ee_positions[:, 0], ee_positions[:, 1], color=colors[i], 
                   linewidth=2, label=strategy, alpha=0.8)
            
            # Mark start and end points
            ax.scatter(ee_positions[0, 0], ee_positions[0, 1], 
                      color=colors[i], s=100, marker='o', zorder=5)
            ax.scatter(ee_positions[-1, 0], ee_positions[-1, 1], 
                      color=colors[i], s=100, marker='s', zorder=5)
        
        ax.set_title('End-Effector Paths', fontweight='bold')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Strategy performance comparison
        ax = axes[1, 1]
        strategies = list(trajectory_results.keys())
        
        # Calculate metrics for each strategy
        max_conditions = []
        mean_conditions = []
        min_manipulabilities = []
        
        for strategy in strategies:
            result = trajectory_results[strategy]
            if 'condition_numbers' in result:
                max_conditions.append(np.max(result['condition_numbers']))
                mean_conditions.append(np.mean(result['condition_numbers']))
                min_manipulabilities.append(np.min(result['manipulabilities']))
            else:
                max_conditions.append(0)
                mean_conditions.append(0)
                min_manipulabilities.append(0)
        
        x = np.arange(len(strategies))
        width = 0.25
        
        ax.bar(x - width, max_conditions, width, label='Max Condition Number', alpha=0.8)
        ax.bar(x, mean_conditions, width, label='Mean Condition Number', alpha=0.8)
        
        ax2 = ax.twinx()
        ax2.bar(x + width, min_manipulabilities, width, label='Min Manipulability', 
               alpha=0.8, color='green')
        
        ax.set_title('Strategy Performance Metrics', fontweight='bold')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Condition Number')
        ax2.set_ylabel('Manipulability')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Get the directory of the current script
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, 'singularity_avoidance.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"📊 Singularity avoidance plot saved as '{save_path}'")
        plt.close()
    
    def demonstrate_dexterity_optimization(self):
        """Demonstrate dexterity optimization and analysis."""
        logger.info("\n🎯 Demonstrating Dexterity Optimization...")
        
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available, skipping optimization")
            return
        
        # Define target end-effector position
        num_joints = len(self.joint_limits)
        target_position = np.array([0.5, 0.3, 0.4])  # Example target
        
        logger.info(f"Optimizing dexterity for target position: {target_position}")
        
        # Different optimization objectives
        objectives = {
            'Maximize_Manipulability': self.maximize_manipulability_objective,
            'Minimize_Condition_Number': self.minimize_condition_objective,
            'Maximize_Min_Singular_Value': self.maximize_min_sv_objective,
        }
        
        optimization_results = {}
        
        for obj_name, obj_func in objectives.items():
            logger.info(f"  Optimizing: {obj_name}")
            
            try:
                result = self.optimize_configuration(target_position, obj_func)
                optimization_results[obj_name] = result
                
                logger.info(f"    Optimal config: {result['optimal_config']}")
                logger.info(f"    Objective value: {result['objective_value']:.6f}")
                
            except Exception as e:
                logger.error(f"Optimization {obj_name} failed: {e}")
        
        # Analyze and compare results
        self.analyze_dexterity_results(optimization_results, target_position)
        
        # Plot optimization results
        self.plot_dexterity_optimization(optimization_results, target_position)
        
        logger.info("✅ Dexterity optimization demonstration complete")
        
        return optimization_results

    def maximize_manipulability_objective(self, config, target_position):
        """Objective function to maximize manipulability."""
        # Check if configuration reaches target
        T = self.robot.forward_kinematics(config)
        ee_position = T[:3, 3]
        position_error = np.linalg.norm(ee_position - target_position)
        
        if position_error > 0.1:  # Position constraint
            return 1e6  # Large penalty
        
        # Calculate manipulability (negative for minimization)
        J = self.robot.jacobian(config)
        manipulability = np.sqrt(np.linalg.det(J @ J.T))
        
        return -manipulability  # Negative for maximization
    
    def minimize_condition_objective(self, config, target_position):
        """Objective function to minimize condition number."""
        # Check if configuration reaches target
        T = self.robot.forward_kinematics(config)
        ee_position = T[:3, 3]
        position_error = np.linalg.norm(ee_position - target_position)
        
        if position_error > 0.1:  # Position constraint
            return 1e6  # Large penalty
        
        # Calculate condition number
        J = self.robot.jacobian(config)
        condition_number = np.linalg.cond(J)
        
        return condition_number
    
    def maximize_min_sv_objective(self, config, target_position):
        """Objective function to maximize minimum singular value."""
        # Check if configuration reaches target
        T = self.robot.forward_kinematics(config)
        ee_position = T[:3, 3]
        position_error = np.linalg.norm(ee_position - target_position)
        
        if position_error > 0.1:  # Position constraint
            return 1e6  # Large penalty
        
        # Calculate minimum singular value (negative for maximization)
        J = self.robot.jacobian(config)
        U, s, Vt = svd(J)
        min_sv = np.min(s)
        
        return -min_sv  # Negative for maximization
    
    def optimize_configuration(self, target_position, objective_func):
        """Optimize robot configuration for given objective."""
        num_joints = len(self.joint_limits)
        
        # Initial guess (middle of joint range)
        initial_guess = np.array([
            (self.joint_limits[i, 0] + self.joint_limits[i, 1]) / 2
            for i in range(num_joints)
        ])
        
        # Bounds for optimization
        bounds = [(self.joint_limits[i, 0], self.joint_limits[i, 1]) 
                 for i in range(num_joints)]
        
        # Optimize
        result = minimize(
            lambda config: objective_func(config, target_position),
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        optimal_config = result.x
        
        # Calculate metrics for optimal configuration
        T = self.robot.forward_kinematics(optimal_config)
        J = self.robot.jacobian(optimal_config)
        U, s, Vt = svd(J)
        
        return {
            'optimal_config': optimal_config,
            'objective_value': result.fun,
            'success': result.success,
            'ee_position': T[:3, 3],
            'jacobian': J,
            'manipulability': np.sqrt(np.linalg.det(J @ J.T)),
            'condition_number': np.linalg.cond(J),
            'min_singular_value': np.min(s)
        }
    
    def analyze_dexterity_results(self, optimization_results, target_position):
        """Analyze dexterity optimization results."""
        logger.info("\n📊 Dexterity Optimization Results Analysis:")
        
        for obj_name, result in optimization_results.items():
            logger.info(f"  {obj_name}:")
            logger.info(f"    Success: {result['success']}")
            logger.info(f"    EE position: {result['ee_position']}")
            logger.info(f"    Position error: {np.linalg.norm(result['ee_position'] - target_position):.6f} m")
            logger.info(f"    Manipulability: {result['manipulability']:.6f}")
            logger.info(f"    Condition number: {result['condition_number']:.2f}")
            logger.info(f"    Min singular value: {result['min_singular_value']:.6f}")
    
    def plot_dexterity_optimization(self, optimization_results, target_position):
        """Plot dexterity optimization results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Dexterity Optimization Results', fontsize=16, fontweight='bold')
        
        obj_names = list(optimization_results.keys())
        colors = ['blue', 'red', 'green']
        
        # Configuration comparison
        ax = axes[0, 0]
        for i, (obj_name, result) in enumerate(optimization_results.items()):
            config = result['optimal_config']
            ax.bar(np.arange(len(config)) + i*0.25, config, width=0.25, 
                  label=obj_name.replace('_', ' '), color=colors[i], alpha=0.8)
        
        ax.set_title('Optimal Joint Configurations', fontweight='bold')
        ax.set_xlabel('Joint Index')
        ax.set_ylabel('Joint Angle (rad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Performance metrics comparison
        ax = axes[0, 1]
        manipulabilities = [result['manipulability'] for result in optimization_results.values()]
        condition_numbers = [result['condition_number'] for result in optimization_results.values()]
        min_svs = [result['min_singular_value'] for result in optimization_results.values()]
        
        x = np.arange(len(obj_names))
        width = 0.25
        
        ax.bar(x - width, manipulabilities, width, label='Manipulability', alpha=0.8, color='skyblue')
        ax.bar(x, [1/cn for cn in condition_numbers], width, label='1/Condition Number', alpha=0.8, color='lightcoral')
        ax.bar(x + width, min_svs, width, label='Min Singular Value', alpha=0.8, color='lightgreen')
        
        ax.set_title('Performance Metrics Comparison', fontweight='bold')
        ax.set_xlabel('Optimization Objective')
        ax.set_ylabel('Metric Value')
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace('_', ' ') for name in obj_names], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # End-effector positions
        ax = axes[1, 0]
        ee_positions = [result['ee_position'] for result in optimization_results.values()]
        
        # Plot target position
        ax.scatter(target_position[0], target_position[1], color='black', s=200, marker='*', 
                  label='Target', zorder=5)
        
        for i, (obj_name, ee_pos) in enumerate(zip(obj_names, ee_positions)):
            ax.scatter(ee_pos[0], ee_pos[1], color=colors[i], s=100, 
                      label=obj_name.replace('_', ' '), alpha=0.8)
        
        ax.set_title('End-Effector Positions', fontweight='bold')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Singular values comparison
        ax = axes[1, 1]
        for i, (obj_name, result) in enumerate(optimization_results.items()):
            J = result['jacobian']
            U, s, Vt = svd(J)
            ax.bar(np.arange(len(s)) + i*0.25, s, width=0.25, 
                  label=obj_name.replace('_', ' '), color=colors[i], alpha=0.8)
        
        ax.set_title('Singular Values Comparison', fontweight='bold')
        ax.set_xlabel('Singular Value Index')
        ax.set_ylabel('Singular Value')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Get the directory of the current script
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, 'dexterity_optimization.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"📊 Dexterity optimization plot saved as '{save_path}'")
        plt.close()
    
    def demonstrate_real_time_monitoring(self):
        """Demonstrate real-time singularity monitoring."""
        logger.info("\n🎯 Demonstrating Real-Time Singularity Monitoring...")
        
        # Simulate a trajectory with real-time monitoring
        num_joints = len(self.joint_limits)
        
        # Create a trajectory that passes near singularities
        trajectory_points = []
        monitoring_data = []
        
        # Generate trajectory through workspace
        num_steps = 100
        for i in range(num_steps):
            # Create a trajectory that explores different regions
            t = i / (num_steps - 1)
            
            config = np.zeros(num_joints)
            for j in range(num_joints):
                # Sinusoidal motion with different frequencies
                amplitude = (self.joint_limits[j, 1] - self.joint_limits[j, 0]) * 0.4
                center = (self.joint_limits[j, 1] + self.joint_limits[j, 0]) / 2
                frequency = 1 + j * 0.5
                config[j] = center + amplitude * np.sin(2 * np.pi * frequency * t)
            
            trajectory_points.append(config)
            
            # Monitor singularities in real-time
            monitoring_data.append(self.monitor_singularity_real_time(config))
        
        # Analyze monitoring results
        self.analyze_monitoring_results(trajectory_points, monitoring_data)
        
        # Plot real-time monitoring
        self.plot_real_time_monitoring(trajectory_points, monitoring_data)
        
        logger.info("✅ Real-time singularity monitoring demonstration complete")
        
        return trajectory_points, monitoring_data
    
    def monitor_singularity_real_time(self, config):
        """Monitor singularities for a single configuration in real-time."""
        # Fast singularity metrics computation
        J = self.robot.jacobian(config)
        
        # Quick condition number estimation
        condition_number = np.linalg.cond(J)
        
        # Quick manipulability
        manipulability = np.sqrt(np.linalg.det(J @ J.T))
        
        # Singularity flags
        is_singular = manipulability < 1e-6
        is_near_singular = condition_number > 100
        
        # Distance to singularity estimate
        U, s, Vt = svd(J)
        min_sv = np.min(s)
        distance_to_singularity = min_sv
        
        return {
            'condition_number': condition_number,
            'manipulability': manipulability,
            'min_singular_value': min_sv,
            'is_singular': is_singular,
            'is_near_singular': is_near_singular,
            'distance_to_singularity': distance_to_singularity,
            'safety_level': self.compute_safety_level(condition_number, min_sv)
        }
    
    def compute_safety_level(self, condition_number, min_sv):
        """Compute safety level based on singularity metrics."""
        # Define safety thresholds
        if min_sv < 1e-4 or condition_number > 1000:
            return 'CRITICAL'  # Very close to singularity
        elif min_sv < 1e-2 or condition_number > 100:
            return 'WARNING'   # Near singularity
        elif min_sv < 0.1 or condition_number > 10:
            return 'CAUTION'   # Approaching singularity
        else:
            return 'SAFE'      # Safe operation
    
    def analyze_monitoring_results(self, trajectory_points, monitoring_data):
        """Analyze real-time monitoring results."""
        logger.info("\n📊 Real-Time Monitoring Analysis:")
        
        safety_levels = [data['safety_level'] for data in monitoring_data]
        
        # Count safety level occurrences
        safety_counts = {
            'SAFE': safety_levels.count('SAFE'),
            'CAUTION': safety_levels.count('CAUTION'),
            'WARNING': safety_levels.count('WARNING'),
            'CRITICAL': safety_levels.count('CRITICAL')
        }
        
        total_points = len(monitoring_data)
        
        logger.info(f"  Safety level distribution:")
        for level, count in safety_counts.items():
            percentage = 100 * count / total_points
            logger.info(f"    {level}: {count}/{total_points} ({percentage:.1f}%)")
        
        # Find critical regions
        critical_indices = [i for i, data in enumerate(monitoring_data) 
                           if data['safety_level'] == 'CRITICAL']
        
        if critical_indices:
            logger.info(f"  Critical singularity regions found at trajectory points: {critical_indices}")
        else:
            logger.info("  No critical singularity regions detected")
        
        # Performance statistics
        condition_numbers = [data['condition_number'] for data in monitoring_data]
        manipulabilities = [data['manipulability'] for data in monitoring_data]
        
        logger.info(f"  Condition number range: [{np.min(condition_numbers):.2f}, {np.max(condition_numbers):.2f}]")
        logger.info(f"  Manipulability range: [{np.min(manipulabilities):.6f}, {np.max(manipulabilities):.6f}]")
    
    def plot_real_time_monitoring(self, trajectory_points, monitoring_data):
        """Plot real-time monitoring results."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Real-Time Singularity Monitoring', fontsize=16, fontweight='bold')
        
        trajectory_length = len(monitoring_data)
        time_steps = np.arange(trajectory_length)
        
        # Safety level timeline
        ax = axes[0, 0]
        safety_levels = [data['safety_level'] for data in monitoring_data]
        safety_colors = {'SAFE': 'green', 'CAUTION': 'yellow', 'WARNING': 'orange', 'CRITICAL': 'red'}
        
        for i, level in enumerate(safety_levels):
            ax.scatter(i, 1, color=safety_colors[level], s=20, alpha=0.8)
        
        ax.set_title('Safety Level Timeline', fontweight='bold')
        ax.set_xlabel('Trajectory Step')
        ax.set_ylabel('Safety Level')
        ax.set_ylim(0.5, 1.5)
        ax.set_yticks([1])
        ax.set_yticklabels(['Safety'])
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=level) 
                          for level, color in safety_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Condition number evolution
        ax = axes[0, 1]
        condition_numbers = [data['condition_number'] for data in monitoring_data]
        ax.plot(time_steps, condition_numbers, 'b-', linewidth=2, alpha=0.8)
        ax.axhline(y=100, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')
        ax.axhline(y=1000, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
        
        ax.set_title('Condition Number Evolution', fontweight='bold')
        ax.set_xlabel('Trajectory Step')
        ax.set_ylabel('Condition Number')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Manipulability evolution
        ax = axes[1, 0]
        manipulabilities = [data['manipulability'] for data in monitoring_data]
        ax.plot(time_steps, manipulabilities, 'g-', linewidth=2, alpha=0.8)
        ax.axhline(y=1e-6, color='red', linestyle='--', alpha=0.7, label='Singular Threshold')
        ax.axhline(y=1e-2, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')
        
        ax.set_title('Manipulability Evolution', fontweight='bold')
        ax.set_xlabel('Trajectory Step')
        ax.set_ylabel('Manipulability')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Distance to singularity
        ax = axes[1, 1]
        distances = [data['distance_to_singularity'] for data in monitoring_data]
        ax.plot(time_steps, distances, 'purple', linewidth=2, alpha=0.8)
        ax.axhline(y=1e-4, color='red', linestyle='--', alpha=0.7, label='Critical Distance')
        ax.axhline(y=1e-2, color='orange', linestyle='--', alpha=0.7, label='Warning Distance')
        
        ax.set_title('Distance to Singularity', fontweight='bold')
        ax.set_xlabel('Trajectory Step')
        ax.set_ylabel('Min Singular Value')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Joint space trajectory
        ax = axes[2, 0]
        trajectory_array = np.array(trajectory_points)
        num_joints = trajectory_array.shape[1]
        
        for j in range(min(4, num_joints)):  # Plot first 4 joints
            ax.plot(time_steps, trajectory_array[:, j], linewidth=2, label=f'Joint {j+1}', alpha=0.8)
        
        ax.set_title('Joint Space Trajectory', fontweight='bold')
        ax.set_xlabel('Trajectory Step')
        ax.set_ylabel('Joint Angle (rad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # End-effector path with safety coloring
        ax = axes[2, 1]
        ee_positions = []
        
        for config in trajectory_points:
            T = self.robot.forward_kinematics(config)
            ee_positions.append(T[:3, 3])
        
        ee_positions = np.array(ee_positions)
        
        # Color code path by safety level
        for i in range(len(ee_positions) - 1):
            color = safety_colors[safety_levels[i]]
            ax.plot(ee_positions[i:i+2, 0], ee_positions[i:i+2, 1], 
                   color=color, linewidth=3, alpha=0.8)
        
        ax.set_title('End-Effector Path (Safety Colored)', fontweight='bold')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        # Get the directory of the current script
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, 'real_time_monitoring.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"📊 Real-time monitoring plot saved as '{save_path}'")
        plt.close()
    
    def run_complete_demonstration(self):
        """Run the complete intermediate singularity analysis demonstration."""
        logger.info("🚀 Starting Intermediate Singularity Analysis Demonstration")
        logger.info("=" * 70)
        
        try:
            # 1. Basic singularity detection
            singularity_results = self.demonstrate_basic_singularity_detection()
            
            # 2. Manipulability ellipsoid analysis
            ellipsoid_results = self.demonstrate_manipulability_ellipsoid_analysis()
            
            # 3. Workspace analysis
            workspace_results = self.demonstrate_workspace_analysis()
            
            # 4. Singularity avoidance strategies
            avoidance_results = self.demonstrate_singularity_avoidance()
            
            # 5. Dexterity optimization
            if SCIPY_AVAILABLE:
                dexterity_results = self.demonstrate_dexterity_optimization()
            else:
                logger.warning("Skipping dexterity optimization (SciPy not available)")
                dexterity_results = None
            
            # 6. Real-time monitoring
            monitoring_results = self.demonstrate_real_time_monitoring()
            
            # Final summary
            self.print_demonstration_summary()
            
        except Exception as e:
            logger.error(f"❌ Demonstration failed: {e}")
            raise
        
        logger.info("🎉 Intermediate Singularity Analysis Demonstration Complete!")
        logger.info("=" * 70)
    
    def print_demonstration_summary(self):
        """Print a summary of the demonstration results."""
        logger.info("\n📋 Demonstration Summary:")
        logger.info("  ✅ Basic singularity detection and analysis")
        logger.info("  ✅ Manipulability ellipsoid visualization") 
        logger.info("  ✅ Workspace analysis with singularity mapping")
        logger.info("  ✅ Singularity avoidance strategies")
        if SCIPY_AVAILABLE:
            logger.info("  ✅ Dexterity optimization")
        else:
            logger.info("  ⚠️ Dexterity optimization (skipped - SciPy not available)")
        logger.info("  ✅ Real-time singularity monitoring")
        
        logger.info(f"\n📁 Generated Files:")
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        generated_files = [
            'singularity_comparison.png',
            'manipulability_ellipsoids.png', 
            'workspace_analysis.png',
            'boundary_analysis.png',
            'singularity_avoidance.png',
            'dexterity_optimization.png',
            'real_time_monitoring.png'
        ]
        
        for filename in generated_files:
            file_path = os.path.join(script_dir, filename)
            if os.path.exists(file_path):
                logger.info(f"  ✅ {filename}")
            else:
                logger.info(f"  ❌ {filename} (not generated)")


def main():
    """Main function to run the intermediate singularity analysis demonstration."""
    print("🔧 Intermediate Singularity Analysis Demo - ManipulaPy")
    print("=" * 60)
    
    try:
        # Initialize the demonstration
        # Try XArm first, fall back to simple robot if needed
        demo = IntermediateSingularityDemo(use_simple_robot=False)
        
        # Run complete demonstration
        demo.run_complete_demonstration()
        
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