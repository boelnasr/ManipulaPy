#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Basic Dynamics Demo: Fundamental Dynamics Computation and Analysis

This example demonstrates fundamental dynamics operations for robotic manipulators including
mass matrix computation, Coriolis and centrifugal forces, gravity compensation, inverse
and forward dynamics, and comprehensive analysis of dynamic properties.

Usage:
    python dynamics_basic_demo.py

Expected Output:
    - Console output showing dynamics quantities and properties
    - Mass matrix analysis including condition numbers and eigenvalues
    - Coriolis and gravity force computations
    - Inverse and forward dynamics verification
    - Comprehensive matplotlib visualizations saved to files

Author: ManipulaPy Development Team
"""

import numpy as np
import matplotlib
# Set non-interactive backend before importing pyplot
matplotlib.use('Agg')
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
from ManipulaPy.dynamics import ManipulatorDynamics


class DynamicsBasicDemo:
    """
    Comprehensive demonstration of basic dynamics operations for robotic manipulators.
    """
    
    def __init__(self, output_dir=None):
        """Initialize the demo with robot model loading."""
        self.robot = None
        self.dynamics = None
        self.joint_limits = None
        self.n_joints = 0
        # Save figures in the same folder as the script
        if output_dir is None:
            script_dir = Path(__file__).parent
            self.output_dir = script_dir / "dynamics_demo_output"
        else:
            self.output_dir = Path(output_dir)
        
    def run_demo(self):
        """Run the complete dynamics demonstration."""
        print("=" * 70)
        print("   ManipulaPy: Basic Dynamics Demo")
        print("=" * 70)
        print()
        
        # Create output directory in the same folder as the script
        script_dir = Path(__file__).parent
        output_path = script_dir / "dynamics_demo_output"
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {output_path.absolute()}")
        print()
        
        # Step 1: Load robot model
        if not self.load_robot_model():
            return False
            
        # Step 2: Demonstrate mass matrix analysis
        self.demonstrate_mass_matrix_analysis()
        
        # Step 3: Demonstrate Coriolis and centrifugal forces
        self.demonstrate_coriolis_forces()
        
        # Step 4: Demonstrate gravity forces
        self.demonstrate_gravity_forces()
        
        # Step 5: Demonstrate inverse dynamics
        self.demonstrate_inverse_dynamics()
        
        # Step 6: Demonstrate forward dynamics
        self.demonstrate_forward_dynamics()
        
        # Step 7: Demonstrate configuration space analysis
        self.demonstrate_configuration_space_analysis()
        
        # Step 8: Create comprehensive visualizations
        self.create_visualizations()
        
        print("\n" + "=" * 70)
        print("✅ Dynamics demo completed successfully!")
        print(f"📊 Check the 'plots' directory next to this script for generated plots")
        print("=" * 70)
        
        return True
    
    def load_robot_model(self):
        """Load and initialize robot model with dynamics."""
        print("🤖 Loading Robot Model with Dynamics")
        print("-" * 40)
        
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
                return False
        
        try:
            # Process URDF and create robot model
            print(f"⚙️ Processing URDF file...")
            urdf_processor = URDFToSerialManipulator(urdf_file)
            self.robot = urdf_processor.serial_manipulator
            self.dynamics = urdf_processor.dynamics
            self.joint_limits = np.array(self.robot.joint_limits)
            self.n_joints = len(self.joint_limits)
            
            print(f"✅ {robot_name} loaded successfully!")
            print(f"   • Number of joints: {self.n_joints}")
            print(f"   • Dynamics model: {type(self.dynamics).__name__}")
            print(f"   • Mass matrices available: {hasattr(self.dynamics, 'Glist')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading robot model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def demonstrate_mass_matrix_analysis(self):
        """Demonstrate mass matrix computation and analysis."""
        print(f"\n⚙️ Mass Matrix Analysis")
        print("-" * 40)
        
        # Test configurations for mass matrix analysis
        test_configs = {
            "Home (Zero)": np.zeros(self.n_joints),
            "Mid-range": np.array([np.mean(limits) for limits in self.joint_limits]),
            "Random": np.random.uniform(self.joint_limits[:, 0], self.joint_limits[:, 1]),
            "Extended": self._generate_extended_pose(),
            "Folded": self._generate_folded_pose()
        }
        
        self.mass_matrix_results = {}
        
        for config_name, joint_angles in test_configs.items():
            print(f"\n📊 Configuration: {config_name}")
            print(f"   Joint angles: {joint_angles}")
            
            # Compute mass matrix
            start_time = time.time()
            M = self.dynamics.mass_matrix(joint_angles)
            computation_time = time.time() - start_time
            
            # Analyze mass matrix properties
            eigenvalues = np.linalg.eigvals(M)
            condition_number = np.linalg.cond(M)
            determinant = np.linalg.det(M)
            trace = np.trace(M)
            frobenius_norm = np.linalg.norm(M, 'fro')
            
            # Check positive definiteness
            is_positive_definite = np.all(eigenvalues > 0)
            min_eigenvalue = np.min(eigenvalues)
            max_eigenvalue = np.max(eigenvalues)
            
            print(f"   Mass matrix shape: {M.shape}")
            print(f"   Computation time: {computation_time*1000:.2f} ms")
            print(f"   Condition number: {condition_number:.2f}")
            print(f"   Determinant: {determinant:.2e}")
            print(f"   Trace: {trace:.3f}")
            print(f"   Frobenius norm: {frobenius_norm:.3f}")
            print(f"   Eigenvalue range: [{min_eigenvalue:.3f}, {max_eigenvalue:.3f}]")
            print(f"   Positive definite: {is_positive_definite}")
            
            if not is_positive_definite:
                print(f"   ⚠️ Warning: Mass matrix is not positive definite!")
            
            # Store results
            self.mass_matrix_results[config_name] = {
                'joint_angles': joint_angles,
                'mass_matrix': M,
                'eigenvalues': eigenvalues,
                'condition_number': condition_number,
                'determinant': determinant,
                'trace': trace,
                'frobenius_norm': frobenius_norm,
                'min_eigenvalue': min_eigenvalue,
                'max_eigenvalue': max_eigenvalue,
                'is_positive_definite': is_positive_definite,
                'computation_time': computation_time
            }
        
        # Performance summary
        avg_time = np.mean([result['computation_time'] for result in self.mass_matrix_results.values()])
        print(f"\n📈 Mass Matrix Performance Summary:")
        print(f"   Average computation time: {avg_time*1000:.2f} ms")
        print(f"   Frequency capability: {1/avg_time:.0f} Hz")
        
        # Configuration space analysis
        condition_numbers = [result['condition_number'] for result in self.mass_matrix_results.values()]
        print(f"   Condition number range: [{np.min(condition_numbers):.1f}, {np.max(condition_numbers):.1f}]")
    
    def demonstrate_coriolis_forces(self):
        """Demonstrate Coriolis and centrifugal forces computation."""
        print(f"\n🌪️ Coriolis and Centrifugal Forces Analysis")
        print("-" * 50)
        
        # Use configurations from mass matrix analysis
        self.coriolis_results = {}
        
        for config_name, mass_result in self.mass_matrix_results.items():
            joint_angles = mass_result['joint_angles']
            
            # Generate different velocity profiles
            velocity_profiles = {
                "Low velocity": 0.1 * np.random.uniform(-1, 1, self.n_joints),
                "Medium velocity": 0.5 * np.random.uniform(-1, 1, self.n_joints),
                "High velocity": 1.0 * np.random.uniform(-1, 1, self.n_joints),
                "Sinusoidal": 0.3 * np.sin(np.linspace(0, 2*np.pi, self.n_joints))
            }
            
            config_coriolis = {}
            
            print(f"\n🌪️ Configuration: {config_name}")
            
            for vel_name, joint_velocities in velocity_profiles.items():
                # Compute Coriolis forces
                start_time = time.time()
                C = self.dynamics.velocity_quadratic_forces(joint_angles, joint_velocities)
                computation_time = time.time() - start_time
                
                # Analyze Coriolis forces
                coriolis_magnitude = np.linalg.norm(C)
                max_coriolis = np.max(np.abs(C))
                
                print(f"   {vel_name}:")
                print(f"     Velocities: {joint_velocities}")
                print(f"     Coriolis forces: {C}")
                print(f"     Magnitude: {coriolis_magnitude:.4f} Nm")
                print(f"     Max component: {max_coriolis:.4f} Nm")
                print(f"     Computation time: {computation_time*1000:.2f} ms")
                
                config_coriolis[vel_name] = {
                    'joint_velocities': joint_velocities,
                    'coriolis_forces': C,
                    'magnitude': coriolis_magnitude,
                    'max_component': max_coriolis,
                    'computation_time': computation_time
                }
            
            self.coriolis_results[config_name] = config_coriolis
        
        # Velocity dependency analysis
        print(f"\n📈 Coriolis Forces Analysis:")
        print("   • Coriolis forces are quadratic in joint velocities")
        print("   • They represent coupling between joint motions")
        print("   • Important for high-speed motion control")
    
    def demonstrate_gravity_forces(self):
        """Demonstrate gravity forces computation."""
        print(f"\n🌍 Gravity Forces Analysis")
        print("-" * 30)
        
        # Test different gravity vectors
        gravity_scenarios = {
            "Earth gravity (down)": np.array([0, 0, -9.81]),
            "Earth gravity (up)": np.array([0, 0, 9.81]),
            "Mars gravity": np.array([0, 0, -3.71]),
            "Moon gravity": np.array([0, 0, -1.62]),
            "Zero gravity": np.array([0, 0, 0]),
            "Horizontal gravity": np.array([9.81, 0, 0])
        }
        
        self.gravity_results = {}
        
        for gravity_name, gravity_vector in gravity_scenarios.items():
            print(f"\n🌍 Gravity scenario: {gravity_name}")
            print(f"   Gravity vector: {gravity_vector} m/s²")
            
            scenario_results = {}
            
            for config_name, mass_result in self.mass_matrix_results.items():
                joint_angles = mass_result['joint_angles']
                
                # Compute gravity forces
                start_time = time.time()
                G = self.dynamics.gravity_forces(joint_angles, gravity_vector)
                computation_time = time.time() - start_time
                
                # Analyze gravity forces
                gravity_magnitude = np.linalg.norm(G)
                max_gravity = np.max(np.abs(G))
                
                print(f"     {config_name}:")
                print(f"       Gravity forces: {G}")
                print(f"       Magnitude: {gravity_magnitude:.4f} Nm")
                print(f"       Max component: {max_gravity:.4f} Nm")
                
                scenario_results[config_name] = {
                    'gravity_forces': G,
                    'magnitude': gravity_magnitude,
                    'max_component': max_gravity,
                    'computation_time': computation_time
                }
            
            self.gravity_results[gravity_name] = scenario_results
        
        # Gravity dependency analysis
        print(f"\n📈 Gravity Forces Analysis:")
        print("   • Gravity forces depend only on robot configuration")
        print("   • Independent of joint velocities and accelerations")
        print("   • Critical for static equilibrium and compensation")
    
    def demonstrate_inverse_dynamics(self):
        """Demonstrate inverse dynamics computation."""
        print(f"\n🔄 Inverse Dynamics Analysis")
        print("-" * 35)
        
        self.inverse_dynamics_results = {}
        
        # Use first configuration for detailed analysis
        config_name = list(self.mass_matrix_results.keys())[0]
        joint_angles = self.mass_matrix_results[config_name]['joint_angles']
        
        print(f"📊 Using configuration: {config_name}")
        print(f"   Joint angles: {joint_angles}")
        
        # Generate different motion profiles
        motion_profiles = {
            "Slow motion": {
                'velocities': 0.1 * np.random.uniform(-1, 1, self.n_joints),
                'accelerations': 0.1 * np.random.uniform(-1, 1, self.n_joints)
            },
            "Medium motion": {
                'velocities': 0.5 * np.random.uniform(-1, 1, self.n_joints),
                'accelerations': 0.5 * np.random.uniform(-1, 1, self.n_joints)
            },
            "Fast motion": {
                'velocities': 1.0 * np.random.uniform(-1, 1, self.n_joints),
                'accelerations': 1.0 * np.random.uniform(-1, 1, self.n_joints)
            },
            "Pure acceleration": {
                'velocities': np.zeros(self.n_joints),
                'accelerations': 1.0 * np.random.uniform(-1, 1, self.n_joints)
            }
        }
        
        gravity_vector = np.array([0, 0, -9.81])
        external_forces = np.zeros(6)  # No external forces
        
        for motion_name, motion_data in motion_profiles.items():
            joint_velocities = motion_data['velocities']
            joint_accelerations = motion_data['accelerations']
            
            print(f"\n🔄 Motion profile: {motion_name}")
            print(f"   Velocities: {joint_velocities}")
            print(f"   Accelerations: {joint_accelerations}")
            
            # Compute inverse dynamics
            start_time = time.time()
            required_torques = self.dynamics.inverse_dynamics(
                joint_angles, joint_velocities, joint_accelerations,
                gravity_vector, external_forces
            )
            computation_time = time.time() - start_time
            
            # Analyze torque components
            M = self.mass_matrix_results[config_name]['mass_matrix']
            inertial_torques = M @ joint_accelerations
            
            if motion_name in ["Slow motion", "Medium motion", "Fast motion"]:
                config_key = config_name
                if config_key in self.coriolis_results:
                    vel_key = "Medium velocity" if motion_name == "Medium motion" else "Low velocity"
                    if vel_key in self.coriolis_results[config_key]:
                        coriolis_torques = self.coriolis_results[config_key][vel_key]['coriolis_forces']
                    else:
                        coriolis_torques = np.zeros(self.n_joints)
                else:
                    coriolis_torques = np.zeros(self.n_joints)
            else:
                coriolis_torques = np.zeros(self.n_joints)
            
            gravity_torques = self.gravity_results["Earth gravity (down)"][config_name]['gravity_forces']
            
            # Verify inverse dynamics equation: τ = M(q)q̈ + C(q,q̇) + G(q)
            computed_torques = inertial_torques + coriolis_torques + gravity_torques
            verification_error = np.linalg.norm(required_torques - computed_torques)
            
            print(f"   Required torques: {required_torques}")
            print(f"   Inertial component: {inertial_torques}")
            print(f"   Coriolis component: {coriolis_torques}")
            print(f"   Gravity component: {gravity_torques}")
            print(f"   Total torque magnitude: {np.linalg.norm(required_torques):.4f} Nm")
            print(f"   Max torque: {np.max(np.abs(required_torques)):.4f} Nm")
            print(f"   Computation time: {computation_time*1000:.2f} ms")
            print(f"   Verification error: {verification_error:.2e}")
            
            self.inverse_dynamics_results[motion_name] = {
                'joint_velocities': joint_velocities,
                'joint_accelerations': joint_accelerations,
                'required_torques': required_torques,
                'inertial_torques': inertial_torques,
                'coriolis_torques': coriolis_torques,
                'gravity_torques': gravity_torques,
                'total_magnitude': np.linalg.norm(required_torques),
                'max_torque': np.max(np.abs(required_torques)),
                'computation_time': computation_time,
                'verification_error': verification_error
            }
    
    def demonstrate_forward_dynamics(self):
        """Demonstrate forward dynamics computation."""
        print(f"\n⏩ Forward Dynamics Analysis")
        print("-" * 35)
        
        self.forward_dynamics_results = {}
        
        # Use inverse dynamics results as input torques for forward dynamics
        config_name = list(self.mass_matrix_results.keys())[0]
        joint_angles = self.mass_matrix_results[config_name]['joint_angles']
        gravity_vector = np.array([0, 0, -9.81])
        external_forces = np.zeros(6)
        
        print(f"📊 Using configuration: {config_name}")
        
        for motion_name, inv_dyn_result in self.inverse_dynamics_results.items():
            joint_velocities = inv_dyn_result['joint_velocities']
            applied_torques = inv_dyn_result['required_torques']
            true_accelerations = inv_dyn_result['joint_accelerations']
            
            print(f"\n⏩ Motion profile: {motion_name}")
            print(f"   Applied torques: {applied_torques}")
            print(f"   Current velocities: {joint_velocities}")
            
            # Compute forward dynamics
            start_time = time.time()
            computed_accelerations = self.dynamics.forward_dynamics(
                joint_angles, joint_velocities, applied_torques,
                gravity_vector, external_forces
            )
            computation_time = time.time() - start_time
            
            # Verify forward dynamics (should recover original accelerations)
            acceleration_error = np.linalg.norm(computed_accelerations - true_accelerations)
            
            print(f"   Computed accelerations: {computed_accelerations}")
            print(f"   True accelerations: {true_accelerations}")
            print(f"   Acceleration error: {acceleration_error:.2e}")
            print(f"   Computation time: {computation_time*1000:.2f} ms")
            
            # Analyze acceleration magnitude
            accel_magnitude = np.linalg.norm(computed_accelerations)
            max_accel = np.max(np.abs(computed_accelerations))
            
            print(f"   Acceleration magnitude: {accel_magnitude:.4f} rad/s²")
            print(f"   Max acceleration: {max_accel:.4f} rad/s²")
            
            self.forward_dynamics_results[motion_name] = {
                'applied_torques': applied_torques,
                'joint_velocities': joint_velocities,
                'computed_accelerations': computed_accelerations,
                'true_accelerations': true_accelerations,
                'acceleration_error': acceleration_error,
                'acceleration_magnitude': accel_magnitude,
                'max_acceleration': max_accel,
                'computation_time': computation_time
            }
        
        # Forward dynamics verification summary
        avg_error = np.mean([result['acceleration_error'] for result in self.forward_dynamics_results.values()])
        avg_time = np.mean([result['computation_time'] for result in self.forward_dynamics_results.values()])
        
        print(f"\n📈 Forward Dynamics Verification:")
        print(f"   Average acceleration error: {avg_error:.2e}")
        print(f"   Average computation time: {avg_time*1000:.2f} ms")
        print(f"   Consistency check: {'✅ PASSED' if avg_error < 1e-10 else '❌ FAILED'}")
    
    def demonstrate_configuration_space_analysis(self):
        """Demonstrate dynamics properties across configuration space."""
        print(f"\n🗺️ Configuration Space Dynamics Analysis")
        print("-" * 45)
        
        print("🔍 Sampling configuration space for dynamics analysis...")
        
        # Sample random configurations
        n_samples = 100
        sample_configs = []
        condition_numbers = []
        determinants = []
        traces = []
        gravity_magnitudes = []
        
        start_time = time.time()
        
        for i in range(n_samples):
            # Generate random configuration
            joint_config = np.random.uniform(
                self.joint_limits[:, 0],
                self.joint_limits[:, 1]
            )
            
            try:
                # Compute mass matrix
                M = self.dynamics.mass_matrix(joint_config)
                
                # Compute gravity forces
                G = self.dynamics.gravity_forces(joint_config, [0, 0, -9.81])
                
                # Store properties
                sample_configs.append(joint_config)
                condition_numbers.append(np.linalg.cond(M))
                determinants.append(np.linalg.det(M))
                traces.append(np.trace(M))
                gravity_magnitudes.append(np.linalg.norm(G))
                
            except Exception as e:
                print(f"   ⚠️ Error at sample {i}: {e}")
                continue
        
        sampling_time = time.time() - start_time
        
        # Analyze statistics
        print(f"✅ Analyzed {len(sample_configs)} configurations in {sampling_time:.2f} seconds")
        print(f"\n📊 Configuration Space Statistics:")
        print(f"   Condition number:")
        print(f"     Range: [{np.min(condition_numbers):.2f}, {np.max(condition_numbers):.2f}]")
        print(f"     Mean: {np.mean(condition_numbers):.2f} ± {np.std(condition_numbers):.2f}")
        print(f"   Determinant:")
        print(f"     Range: [{np.min(determinants):.2e}, {np.max(determinants):.2e}]")
        print(f"     Mean: {np.mean(determinants):.2e} ± {np.std(determinants):.2e}")
        print(f"   Trace:")
        print(f"     Range: [{np.min(traces):.3f}, {np.max(traces):.3f}]")
        print(f"     Mean: {np.mean(traces):.3f} ± {np.std(traces):.3f}")
        print(f"   Gravity magnitude:")
        print(f"     Range: [{np.min(gravity_magnitudes):.3f}, {np.max(gravity_magnitudes):.3f}] Nm")
        print(f"     Mean: {np.mean(gravity_magnitudes):.3f} ± {np.std(gravity_magnitudes):.3f} Nm")
        
        # Identify problematic configurations
        high_condition = np.array(condition_numbers) > 100
        n_high_condition = np.sum(high_condition)
        
        print(f"\n⚠️ Configurations with high condition number (>100): {n_high_condition}/{len(condition_numbers)} ({100*n_high_condition/len(condition_numbers):.1f}%)")
        
        self.config_space_results = {
            'sample_configs': sample_configs,
            'condition_numbers': condition_numbers,
            'determinants': determinants,
            'traces': traces,
            'gravity_magnitudes': gravity_magnitudes,
            'sampling_time': sampling_time,
            'n_high_condition': n_high_condition
        }
    
    def create_visualizations(self):
        """Create comprehensive visualization plots and save to files."""
        print(f"\n📊 Creating Comprehensive Dynamics Visualizations (saving to files)")
        print("-" * 70)
        
        try:
            # Create output directory in the same folder as the script
            script_dir = Path(__file__).parent
            plot_dir = script_dir / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            # Create comprehensive figure
            fig = plt.figure(figsize=(20, 16))
            fig.suptitle('ManipulaPy: Basic Dynamics Demo - Comprehensive Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Mass Matrix Properties
            ax1 = plt.subplot(4, 4, 1)
            self._plot_mass_matrix_properties(ax1)
            
            # Plot 2: Mass Matrix Heatmap
            ax2 = plt.subplot(4, 4, 2)
            self._plot_mass_matrix_heatmap(ax2)
            
            # Plot 3: Eigenvalue Analysis
            ax3 = plt.subplot(4, 4, 3)
            self._plot_eigenvalue_analysis(ax3)
            
            # Plot 4: Coriolis Force Magnitude
            ax4 = plt.subplot(4, 4, 4)
            self._plot_coriolis_analysis(ax4)
            
            # Plot 5: Gravity Forces Comparison
            ax5 = plt.subplot(4, 4, 5)
            self._plot_gravity_comparison(ax5)
            
            # Plot 6: Torque Component Analysis
            ax6 = plt.subplot(4, 4, 6)
            self._plot_torque_components(ax6)
            
            # Plot 7: Forward/Inverse Dynamics Verification
            ax7 = plt.subplot(4, 4, 7)
            self._plot_dynamics_verification(ax7)
            
            # Plot 8: Configuration Space Analysis
            ax8 = plt.subplot(4, 4, 8)
            self._plot_configuration_space(ax8)
            
            # Plot 9: Performance Metrics
            ax9 = plt.subplot(4, 4, 9)
            self._plot_performance_metrics(ax9)
            
            # Plot 10: Acceleration Analysis
            ax10 = plt.subplot(4, 4, 10)
            self._plot_acceleration_analysis(ax10)
            
            # Plot 11: Condition Number Distribution
            ax11 = plt.subplot(4, 4, 11)
            self._plot_condition_number_distribution(ax11)
            
            # Plot 12: Gravity vs Configuration
            ax12 = plt.subplot(4, 4, 12)
            self._plot_gravity_vs_configuration(ax12)
            
            # Plot 13: Mass Matrix Diagonal
            ax13 = plt.subplot(4, 4, 13)
            self._plot_mass_matrix_diagonal(ax13)
            
            # Plot 14: Torque Magnitude Comparison
            ax14 = plt.subplot(4, 4, 14)
            self._plot_torque_magnitude_comparison(ax14)
            
            # Plot 15: Dynamic Range Analysis
            ax15 = plt.subplot(4, 4, 15)
            self._plot_dynamic_range_analysis(ax15)
            
            # Plot 16: Computational Performance
            ax16 = plt.subplot(4, 4, 16)
            self._plot_computational_performance(ax16)
            
            plt.tight_layout()
            
            # Save the comprehensive figure to file
            comprehensive_plot_path = plot_dir / "dynamics_comprehensive_analysis.png"
            fig.savefig(str(comprehensive_plot_path), dpi=300, bbox_inches='tight')
            print(f"✅ Saved comprehensive plot to {comprehensive_plot_path}")
            
            # Close the figure to free memory
            plt.close(fig)
            
            # Create individual detailed plots
            self._create_individual_plots(plot_dir)
            
            print("✅ All visualization plots created and saved successfully!")
            print(f"📁 Plots saved in: {plot_dir.absolute()}")
            
        except Exception as e:
            print(f"⚠️ Error creating visualizations: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_individual_plots(self, plot_dir):
        """Create individual detailed plots for each analysis."""
        print("📊 Creating individual detailed plots...")
        
        # 1. Mass Matrix Analysis Plot
        fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))  # Reduced size
        fig1.suptitle('Mass Matrix Analysis', fontsize=14, fontweight='bold')
        
        self._plot_mass_matrix_properties(axes1[0, 0])
        self._plot_mass_matrix_heatmap(axes1[0, 1])
        self._plot_eigenvalue_analysis(axes1[1, 0])
        self._plot_mass_matrix_diagonal(axes1[1, 1])
        
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, 
                          wspace=0.3, hspace=0.4)
        mass_plot_path = plot_dir / "mass_matrix_analysis.png"
        fig1.savefig(str(mass_plot_path), dpi=200, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close(fig1)
        print(f"   ✅ Mass matrix analysis: {mass_plot_path}")
        
        # 2. Forces Analysis Plot
        fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
        fig2.suptitle('Forces Analysis', fontsize=14, fontweight='bold')
        
        self._plot_coriolis_analysis(axes2[0, 0])
        self._plot_gravity_comparison(axes2[0, 1])
        self._plot_torque_components(axes2[1, 0])
        self._plot_torque_magnitude_comparison(axes2[1, 1])
        
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, 
                          wspace=0.3, hspace=0.4)
        forces_plot_path = plot_dir / "forces_analysis.png"
        fig2.savefig(str(forces_plot_path), dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig2)
        print(f"   ✅ Forces analysis: {forces_plot_path}")
        
        # 3. Dynamics Verification Plot
        fig3, axes3 = plt.subplots(2, 2, figsize=(10, 8))
        fig3.suptitle('Dynamics Verification', fontsize=14, fontweight='bold')
        
        self._plot_dynamics_verification(axes3[0, 0])
        self._plot_acceleration_analysis(axes3[0, 1])
        self._plot_performance_metrics(axes3[1, 0])
        self._plot_computational_performance(axes3[1, 1])
        
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, 
                          wspace=0.3, hspace=0.4)
        verification_plot_path = plot_dir / "dynamics_verification.png"
        fig3.savefig(str(verification_plot_path), dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig3)
        print(f"   ✅ Dynamics verification: {verification_plot_path}")
        
        # 4. Configuration Space Analysis Plot
        fig4, axes4 = plt.subplots(2, 2, figsize=(10, 8))
        fig4.suptitle('Configuration Space Analysis', fontsize=14, fontweight='bold')
        
        self._plot_configuration_space(axes4[0, 0])
        self._plot_condition_number_distribution(axes4[0, 1])
        self._plot_gravity_vs_configuration(axes4[1, 0])
        self._plot_dynamic_range_analysis(axes4[1, 1])
        
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, 
                          wspace=0.3, hspace=0.4)
        config_space_plot_path = plot_dir / "configuration_space_analysis.png"
        fig4.savefig(str(config_space_plot_path), dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig4)
        print(f"   ✅ Configuration space analysis: {config_space_plot_path}")
        
        # 5. Save data summary to text file
        self._save_data_summary(plot_dir)
    
    def _save_data_summary(self, plot_dir):
        """Save numerical results summary to text file."""
        summary_path = plot_dir / "dynamics_analysis_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("ManipulaPy: Basic Dynamics Demo - Analysis Summary\n")
            f.write("=" * 60 + "\n\n")
            
            # Mass Matrix Results
            f.write("MASS MATRIX ANALYSIS\n")
            f.write("-" * 30 + "\n")
            for config_name, result in self.mass_matrix_results.items():
                f.write(f"\nConfiguration: {config_name}\n")
                f.write(f"  Joint angles: {result['joint_angles']}\n")
                f.write(f"  Condition number: {result['condition_number']:.2f}\n")
                f.write(f"  Determinant: {result['determinant']:.2e}\n")
                f.write(f"  Trace: {result['trace']:.3f}\n")
                f.write(f"  Eigenvalue range: [{result['min_eigenvalue']:.3f}, {result['max_eigenvalue']:.3f}]\n")
                f.write(f"  Positive definite: {result['is_positive_definite']}\n")
                f.write(f"  Computation time: {result['computation_time']*1000:.2f} ms\n")
            
            # Performance Summary
            f.write(f"\nPERFORMANCE SUMMARY\n")
            f.write("-" * 30 + "\n")
            mass_times = [r['computation_time'] for r in self.mass_matrix_results.values()]
            inv_dyn_times = [r['computation_time'] for r in self.inverse_dynamics_results.values()]
            fwd_dyn_times = [r['computation_time'] for r in self.forward_dynamics_results.values()]
            
            f.write(f"Average computation times:\n")
            f.write(f"  Mass Matrix: {np.mean(mass_times)*1000:.2f} ms ({1/np.mean(mass_times):.0f} Hz)\n")
            f.write(f"  Inverse Dynamics: {np.mean(inv_dyn_times)*1000:.2f} ms ({1/np.mean(inv_dyn_times):.0f} Hz)\n")
            f.write(f"  Forward Dynamics: {np.mean(fwd_dyn_times)*1000:.2f} ms ({1/np.mean(fwd_dyn_times):.0f} Hz)\n")
            
            # Configuration Space Analysis
            f.write(f"\nCONFIGURATION SPACE STATISTICS\n")
            f.write("-" * 30 + "\n")
            cs_results = self.config_space_results
            f.write(f"Samples analyzed: {len(cs_results['sample_configs'])}\n")
            f.write(f"Condition number range: [{np.min(cs_results['condition_numbers']):.2f}, {np.max(cs_results['condition_numbers']):.2f}]\n")
            f.write(f"Condition number mean: {np.mean(cs_results['condition_numbers']):.2f} ± {np.std(cs_results['condition_numbers']):.2f}\n")
            f.write(f"High condition configurations: {cs_results['n_high_condition']}/{len(cs_results['condition_numbers'])} ({100*cs_results['n_high_condition']/len(cs_results['condition_numbers']):.1f}%)\n")
            
            # Verification Results
            f.write(f"\nVERIFICATION RESULTS\n")
            f.write("-" * 30 + "\n")
            avg_error = np.mean([result['acceleration_error'] for result in self.forward_dynamics_results.values()])
            f.write(f"Forward/Inverse dynamics consistency: {avg_error:.2e}\n")
            f.write(f"Verification status: {'PASSED' if avg_error < 1e-10 else 'FAILED'}\n")
        
        print(f"   ✅ Data summary: {summary_path}")
    
    def _plot_mass_matrix_properties(self, ax):
        """Plot mass matrix properties comparison."""
        configs = list(self.mass_matrix_results.keys())
        condition_numbers = [self.mass_matrix_results[config]['condition_number'] for config in configs]
        determinants = [self.mass_matrix_results[config]['determinant'] for config in configs]
        traces = [self.mass_matrix_results[config]['trace'] for config in configs]
        
        x = np.arange(len(configs))
        width = 0.25
        
        # Normalize for visualization
        norm_cond = np.array(condition_numbers) / np.max(condition_numbers)
        norm_det = np.array(determinants) / np.max(determinants)
        norm_trace = np.array(traces) / np.max(traces)
        
        ax.bar(x - width, norm_cond, width, label='Condition Number (norm)', alpha=0.8)
        ax.bar(x, norm_det, width, label='Determinant (norm)', alpha=0.8)
        ax.bar(x + width, norm_trace, width, label='Trace (norm)', alpha=0.8)
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Normalized Value')
        ax.set_title('Mass Matrix Properties')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_mass_matrix_heatmap(self, ax):
        """Plot mass matrix heatmap."""
        # Use the first configuration
        config = list(self.mass_matrix_results.keys())[0]
        M = self.mass_matrix_results[config]['mass_matrix']
        
        im = ax.imshow(M, cmap='viridis', aspect='auto')
        ax.set_title(f'Mass Matrix ({config})')
        ax.set_xlabel('Joint Index')
        ax.set_ylabel('Joint Index')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add grid
        ax.set_xticks(range(M.shape[1]))
        ax.set_yticks(range(M.shape[0]))
        ax.grid(True, alpha=0.3)
    
    def _plot_eigenvalue_analysis(self, ax):
        """Plot eigenvalue analysis."""
        configs = list(self.mass_matrix_results.keys())
        
        for config in configs:
            eigenvalues = self.mass_matrix_results[config]['eigenvalues']
            ax.plot(eigenvalues, 'o-', label=config, alpha=0.8)
        
        ax.set_xlabel('Eigenvalue Index')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('Mass Matrix Eigenvalues')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Only use log scale if all eigenvalues are positive
        min_eigenvalue = min([min(self.mass_matrix_results[config]['eigenvalues']) for config in configs])
        if min_eigenvalue > 0:
            ax.set_yscale('log')
        else:
            print(f"   ⚠️ Warning: Negative eigenvalues detected, using linear scale")
    
    def _plot_coriolis_analysis(self, ax):
        """Plot Coriolis force magnitude analysis."""
        if not hasattr(self, 'coriolis_results'):
            ax.text(0.5, 0.5, 'No Coriolis\nresults available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Coriolis Force Analysis')
            return
        
        configs = list(self.coriolis_results.keys())
        velocities = list(self.coriolis_results[configs[0]].keys())
        
        x = np.arange(len(velocities))
        width = 0.8 / len(configs)
        
        for i, config in enumerate(configs):
            magnitudes = [self.coriolis_results[config][vel]['magnitude'] for vel in velocities]
            ax.bar(x + i*width, magnitudes, width, label=config, alpha=0.8)
        
        ax.set_xlabel('Velocity Profile')
        ax.set_ylabel('Coriolis Force Magnitude (Nm)')
        ax.set_title('Coriolis Forces vs Velocity')
        ax.set_xticks(x + width * (len(configs)-1)/2)
        ax.set_xticklabels(velocities, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_gravity_comparison(self, ax):
        """Plot gravity forces comparison."""
        gravity_scenarios = list(self.gravity_results.keys())
        configs = list(self.gravity_results[gravity_scenarios[0]].keys())
        
        # Use first configuration
        config = configs[0]
        magnitudes = [self.gravity_results[scenario][config]['magnitude'] for scenario in gravity_scenarios]
        
        bars = ax.bar(gravity_scenarios, magnitudes, alpha=0.8, color='lightcoral')
        ax.set_ylabel('Gravity Force Magnitude (Nm)')
        ax.set_title(f'Gravity Forces ({config})')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, magnitudes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_torque_components(self, ax):
        """Plot torque component analysis."""
        motions = list(self.inverse_dynamics_results.keys())
        
        # Stack the torque components
        inertial = [np.linalg.norm(self.inverse_dynamics_results[motion]['inertial_torques']) for motion in motions]
        coriolis = [np.linalg.norm(self.inverse_dynamics_results[motion]['coriolis_torques']) for motion in motions]
        gravity = [np.linalg.norm(self.inverse_dynamics_results[motion]['gravity_torques']) for motion in motions]
        
        x = np.arange(len(motions))
        
        ax.bar(x, inertial, label='Inertial', alpha=0.8)
        ax.bar(x, coriolis, bottom=inertial, label='Coriolis', alpha=0.8)
        ax.bar(x, gravity, bottom=np.array(inertial)+np.array(coriolis), label='Gravity', alpha=0.8)
        
        ax.set_xlabel('Motion Profile')
        ax.set_ylabel('Torque Magnitude (Nm)')
        ax.set_title('Torque Component Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(motions, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_dynamics_verification(self, ax):
        """Plot forward/inverse dynamics verification."""
        motions = list(self.forward_dynamics_results.keys())
        errors = [self.forward_dynamics_results[motion]['acceleration_error'] for motion in motions]
        
        bars = ax.bar(motions, errors, alpha=0.8, color='green')
        ax.set_ylabel('Acceleration Error')
        ax.set_title('Forward/Inverse Dynamics Verification')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Only use log scale if all errors are positive
        min_error = min(errors)
        if min_error > 0:
            ax.set_yscale('log')
        else:
            print(f"   ⚠️ Warning: Zero or negative errors detected, using linear scale")
        
        # Add value labels
        for bar, value in zip(bars, errors):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1e}', ha='center', va='bottom', fontsize=8)
    
    def _plot_configuration_space(self, ax):
        """Plot configuration space analysis."""
        condition_numbers = self.config_space_results['condition_numbers']
        gravity_magnitudes = self.config_space_results['gravity_magnitudes']
        
        ax.scatter(condition_numbers, gravity_magnitudes, alpha=0.6, s=20)
        ax.set_xlabel('Condition Number')
        ax.set_ylabel('Gravity Magnitude (Nm)')
        ax.set_title('Configuration Space Analysis')
        ax.grid(True, alpha=0.3)
        
        # Only use log scale if all condition numbers are positive
        min_condition = min(condition_numbers)
        if min_condition > 0:
            ax.set_xscale('log')
        else:
            print(f"   ⚠️ Warning: Zero or negative condition numbers detected, using linear scale")
    
    def _plot_performance_metrics(self, ax):
        """Plot performance metrics summary."""
        # Collect timing data
        mass_times = [r['computation_time']*1000 for r in self.mass_matrix_results.values()]
        inv_dyn_times = [r['computation_time']*1000 for r in self.inverse_dynamics_results.values()]
        fwd_dyn_times = [r['computation_time']*1000 for r in self.forward_dynamics_results.values()]
        
        metrics = {
            'Mass Matrix': np.mean(mass_times),
            'Inverse Dynamics': np.mean(inv_dyn_times),
            'Forward Dynamics': np.mean(fwd_dyn_times)
        }
        
        bars = ax.bar(metrics.keys(), metrics.values(), alpha=0.8, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax.set_ylabel('Computation Time (ms)')
        ax.set_title('Performance Metrics')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, metrics.values()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_acceleration_analysis(self, ax):
        """Plot acceleration analysis."""
        motions = list(self.forward_dynamics_results.keys())
        magnitudes = [self.forward_dynamics_results[motion]['acceleration_magnitude'] for motion in motions]
        max_accels = [self.forward_dynamics_results[motion]['max_acceleration'] for motion in motions]
        
        x = np.arange(len(motions))
        width = 0.35
        
        ax.bar(x - width/2, magnitudes, width, label='RMS Acceleration', alpha=0.8)
        ax.bar(x + width/2, max_accels, width, label='Max Acceleration', alpha=0.8)
        
        ax.set_xlabel('Motion Profile')
        ax.set_ylabel('Acceleration (rad/s²)')
        ax.set_title('Acceleration Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(motions, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_condition_number_distribution(self, ax):
        """Plot condition number distribution."""
        condition_numbers = self.config_space_results['condition_numbers']
        
        ax.hist(condition_numbers, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Condition Number')
        ax.set_ylabel('Frequency')
        ax.set_title('Condition Number Distribution')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_cond = np.mean(condition_numbers)
        ax.axvline(mean_cond, color='red', linestyle='--', label=f'Mean: {mean_cond:.1f}')
        ax.legend()
    
    def _plot_gravity_vs_configuration(self, ax):
        """Plot gravity magnitude vs configuration."""
        gravity_magnitudes = self.config_space_results['gravity_magnitudes']
        
        ax.plot(gravity_magnitudes, 'o', alpha=0.6, markersize=3)
        ax.set_xlabel('Configuration Index')
        ax.set_ylabel('Gravity Magnitude (Nm)')
        ax.set_title('Gravity Forces Across Configurations')
        ax.grid(True, alpha=0.3)
        
        # Add mean line
        mean_gravity = np.mean(gravity_magnitudes)
        ax.axhline(mean_gravity, color='red', linestyle='--', label=f'Mean: {mean_gravity:.2f}')
        ax.legend()
    
    def _plot_mass_matrix_diagonal(self, ax):
        """Plot mass matrix diagonal elements."""
        configs = list(self.mass_matrix_results.keys())
        
        for config in configs:
            M = self.mass_matrix_results[config]['mass_matrix']
            diagonal = np.diag(M)
            ax.plot(diagonal, 'o-', label=config, alpha=0.8)
        
        ax.set_xlabel('Joint Index')
        ax.set_ylabel('Diagonal Element (kg⋅m²)')
        ax.set_title('Mass Matrix Diagonal Elements')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_torque_magnitude_comparison(self, ax):
        """Plot torque magnitude comparison."""
        motions = list(self.inverse_dynamics_results.keys())
        magnitudes = [self.inverse_dynamics_results[motion]['total_magnitude'] for motion in motions]
        max_torques = [self.inverse_dynamics_results[motion]['max_torque'] for motion in motions]
        
        x = np.arange(len(motions))
        width = 0.35
        
        ax.bar(x - width/2, magnitudes, width, label='RMS Torque', alpha=0.8)
        ax.bar(x + width/2, max_torques, width, label='Max Torque', alpha=0.8)
        
        ax.set_xlabel('Motion Profile')
        ax.set_ylabel('Torque (Nm)')
        ax.set_title('Torque Magnitude Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(motions, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_dynamic_range_analysis(self, ax):
        """Plot dynamic range analysis."""
        configs = list(self.mass_matrix_results.keys())
        eigenvalue_ranges = []
        
        for config in configs:
            eigenvalues = self.mass_matrix_results[config]['eigenvalues']
            min_eig = np.min(eigenvalues)
            max_eig = np.max(eigenvalues)
            if min_eig > 0:
                dynamic_range = max_eig / min_eig
            else:
                dynamic_range = np.inf  # Handle negative eigenvalues
            eigenvalue_ranges.append(dynamic_range)
        
        bars = ax.bar(configs, eigenvalue_ranges, alpha=0.8, color='purple')
        ax.set_ylabel('Dynamic Range (λ_max/λ_min)')
        ax.set_title('Mass Matrix Dynamic Range')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Only use log scale if all ranges are finite and positive
        finite_ranges = [r for r in eigenvalue_ranges if np.isfinite(r) and r > 0]
        if len(finite_ranges) == len(eigenvalue_ranges) and min(finite_ranges) > 0:
            ax.set_yscale('log')
        else:
            print(f"   ⚠️ Warning: Infinite or invalid dynamic ranges detected, using linear scale")
        
        # Add value labels
        for bar, value in zip(bars, eigenvalue_ranges):
            height = bar.get_height()
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}', ha='center', va='bottom', fontsize=8)
            else:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
                       'inf', ha='center', va='center', fontsize=8)
    
    def _plot_computational_performance(self, ax):
        """Plot computational performance summary."""
        # Calculate frequencies
        mass_freq = 1 / np.mean([r['computation_time'] for r in self.mass_matrix_results.values()])
        inv_dyn_freq = 1 / np.mean([r['computation_time'] for r in self.inverse_dynamics_results.values()])
        fwd_dyn_freq = 1 / np.mean([r['computation_time'] for r in self.forward_dynamics_results.values()])
        
        frequencies = {
            'Mass Matrix': mass_freq,
            'Inverse Dynamics': inv_dyn_freq,
            'Forward Dynamics': fwd_dyn_freq
        }
        
        bars = ax.bar(frequencies.keys(), frequencies.values(), alpha=0.8, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Computational Performance')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, frequencies.values()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                   f'{value:.0f}', ha='center', va='bottom', fontsize=8)
    
    # Helper methods
    def _generate_extended_pose(self):
        """Generate an extended arm pose."""
        # Move joints towards their positive limits (extended configuration)
        extended = np.zeros(self.n_joints)
        for i in range(self.n_joints):
            if self.joint_limits[i, 1] > 0:
                extended[i] = 0.7 * self.joint_limits[i, 1]
            else:
                extended[i] = 0.3 * self.joint_limits[i, 1]
        return extended
    
    def _generate_folded_pose(self):
        """Generate a folded arm pose."""
        # Move joints towards their negative limits (folded configuration)
        folded = np.zeros(self.n_joints)
        for i in range(self.n_joints):
            if self.joint_limits[i, 0] < 0:
                folded[i] = 0.7 * self.joint_limits[i, 0]
            else:
                folded[i] = 0.3 * self.joint_limits[i, 0]
        return folded


def main():
    """Main function to run the dynamics basic demo."""
    try:
        # Create and run demo
        demo = DynamicsBasicDemo()
        success = demo.run_demo()
        
        if success:
            print("\n🎉 Demo completed successfully!")
            print("📋 Summary of demonstrated concepts:")
            print("   ✅ Mass matrix computation and analysis")
            print("   ✅ Coriolis and centrifugal forces")
            print("   ✅ Gravity forces and compensation")
            print("   ✅ Inverse dynamics (torque calculation)")
            print("   ✅ Forward dynamics (acceleration calculation)")
            print("   ✅ Configuration space analysis")
            print("   ✅ Dynamic properties and conditioning")
            print("   ✅ Performance benchmarking")
            print("   ✅ Comprehensive visualization (saved to files)")
            
            print("\n📚 Key takeaways:")
            print("   • Mass matrix M(q): Configuration-dependent inertia")
            print("   • Coriolis forces C(q,q̇): Velocity-dependent coupling")
            print("   • Gravity forces G(q): Configuration-dependent gravity")
            print("   • Inverse dynamics: τ = M(q)q̈ + C(q,q̇) + G(q)")
            print("   • Forward dynamics: q̈ = M⁻¹(q)[τ - C(q,q̇) - G(q)]")
            print("   • Condition number indicates numerical stability")
            
            print("\n🔗 Next steps:")
            print("   • Try basic_examples/control_basic_demo.py")
            print("   • Explore intermediate_examples/trajectory_planning_intermediate_demo.py")
            print("   • Check out intermediate_examples/simulation_intermediate_demo.py")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()