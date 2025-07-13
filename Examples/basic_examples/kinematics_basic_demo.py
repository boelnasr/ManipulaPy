#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Basic Control Demo - ManipulaPy

This demo showcases the fundamental control capabilities of ManipulaPy including:
- PID Control
- Computed Torque Control  
- PD with Feedforward Control
- Joint Space Control
- Cartesian Space Control
- Controller tuning with Ziegler-Nichols method
- Performance analysis and visualization

Copyright (c) 2025 Mohamed Aboelnar
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # prevents any GUI backend from being invoked

# ManipulaPy imports
try:
    from ManipulaPy.kinematics import SerialManipulator
    from ManipulaPy.dynamics import ManipulatorDynamics
    from ManipulaPy.control import ManipulatorController
    from ManipulaPy.urdf_processor import URDFToSerialManipulator
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

class BasicControlDemo:
    """
    Demonstrates basic control algorithms for robotic manipulators using the built-in XArm robot.
    """
    
    def __init__(self, use_simple_robot=False):
        """
        Initialize the control demo.
        
        Args:
            use_simple_robot: If True, creates a simple 3-DOF robot. 
                             If False (default), uses the built-in XArm robot.
        """
        self.use_simple_robot = use_simple_robot
        self.setup_robot()
        self.setup_controller()
        
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
            # These are conservative values for safe operation
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
            
            # Display joint limit information
            for i, (joint_min, joint_max) in enumerate(self.joint_limits):
                logger.info(f"   Joint {i+1}: [{joint_min:.3f}, {joint_max:.3f}] rad "
                          f"([{np.degrees(joint_min):.1f}, {np.degrees(joint_max):.1f}] deg)")
                          
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
    
    def get_safe_joint_targets(self):
        """Get safe joint target positions within limits for the current robot."""
        num_joints = len(self.joint_limits)
        
        if not self.use_simple_robot:  # XArm robot
            # Conservative targets for XArm within joint limits
            # These are safe positions that don't cause singularities
            safe_targets = {
                6: np.array([np.pi/6, np.pi/8, np.pi/4, np.pi/6, np.pi/8, np.pi/6]),  # 6-DOF
                7: np.array([np.pi/6, np.pi/8, np.pi/4, np.pi/6, np.pi/8, np.pi/6, np.pi/8])  # 7-DOF
            }
            
            if num_joints in safe_targets:
                targets = safe_targets[num_joints]
            else:
                # Generate safe targets automatically
                targets = np.array([
                    (self.joint_limits[i, 0] + self.joint_limits[i, 1]) * 0.2
                    for i in range(num_joints)
                ])
        else:  # Simple robot
            targets = np.array([np.pi/4, np.pi/6, -np.pi/8][:num_joints])
        
        # Ensure targets are within limits
        targets = np.clip(targets, self.joint_limits[:, 0], self.joint_limits[:, 1])
        
        return targets
    
    def get_trajectory_amplitudes(self):
        """Get safe trajectory amplitudes for the current robot."""
        num_joints = len(self.joint_limits)
        
        if not self.use_simple_robot:  # XArm robot
            # Conservative amplitudes to avoid joint limits
            range_factor = 0.3  # Use 30% of joint range
            amplitudes = np.array([
                (self.joint_limits[i, 1] - self.joint_limits[i, 0]) * range_factor
                for i in range(num_joints)
            ])
        else:  # Simple robot
            amplitudes = np.array([np.pi/6, np.pi/8, np.pi/10][:num_joints])
        
        return amplitudes
        
    def setup_urdf_robot(self):
        """Legacy method - now redirects to XArm setup."""
        logger.info("Redirecting to built-in XArm robot...")
        self.setup_xarm_robot()
        
    def setup_controller(self):
        """Initialize the manipulator controller."""
        self.controller = ManipulatorController(self.dynamics)
        logger.info("✅ Controller initialized")
        
    def demonstrate_pid_control(self):
        """Demonstrate PID control with step response analysis."""
        logger.info("\n🎯 Demonstrating PID Control...")
        
        # Control parameters
        num_joints = len(self.joint_limits)
        Kp = np.array([100.0] * num_joints)
        Ki = np.array([10.0] * num_joints)
        Kd = np.array([20.0] * num_joints)
        
        # Simulation parameters
        dt = 0.01  # 10 ms time step
        T_final = 5.0  # 5 second simulation
        time_steps = int(T_final / dt)
        
        # Target positions (step inputs) - use safe targets for current robot
        target_positions = self.get_safe_joint_targets()
        target_velocities = np.zeros(num_joints)
        
        # Initial conditions
        current_positions = np.zeros(num_joints)
        current_velocities = np.zeros(num_joints)
        
        # Data storage
        time_history = []
        position_history = []
        velocity_history = []
        torque_history = []
        error_history = []
        
        logger.info(f"Running PID simulation for {T_final}s with {time_steps} steps...")
        
        for step in range(time_steps):
            current_time = step * dt
            
            # PID control
            control_torques = self.controller.pid_control(
                thetalistd=target_positions,
                dthetalistd=target_velocities,
                thetalist=current_positions,
                dthetalist=current_velocities,
                dt=dt,
                Kp=Kp,
                Ki=Ki,
                Kd=Kd
            )
            
            # Convert to numpy if using GPU
            if GPU_AVAILABLE and hasattr(control_torques, 'get'):
                control_torques = control_torques.get()
            
            # Apply torque limits
            control_torques = np.clip(
                control_torques, 
                self.torque_limits[:, 0], 
                self.torque_limits[:, 1]
            )
            
            # Simple integration (Euler method)
            # In a real system, you would use forward dynamics
            # For demo purposes, we'll use a simplified model
            mass_matrix = np.diag([1.0] * num_joints)  # Simplified
            accelerations = np.linalg.solve(mass_matrix, control_torques)
            
            # Update states
            current_velocities += accelerations * dt
            current_positions += current_velocities * dt
            
            # Apply joint limits
            current_positions = np.clip(
                current_positions,
                self.joint_limits[:, 0],
                self.joint_limits[:, 1]
            )
            
            # Store data
            time_history.append(current_time)
            position_history.append(current_positions.copy())
            velocity_history.append(current_velocities.copy())
            torque_history.append(control_torques.copy())
            error_history.append(target_positions - current_positions)
            
        # Convert to numpy arrays
        time_history = np.array(time_history)
        position_history = np.array(position_history)
        velocity_history = np.array(velocity_history)
        torque_history = np.array(torque_history)
        error_history = np.array(error_history)
        
        # Plot results
        self.plot_pid_results(
            time_history, position_history, velocity_history, 
            torque_history, error_history, target_positions
        )
        
        # Analyze performance
        self.analyze_step_response(time_history, position_history, target_positions)
        
        logger.info("✅ PID control demonstration complete")
        
    def plot_pid_results(self, time_hist, pos_hist, vel_hist, torque_hist, error_hist, targets):
        """Plot PID control results."""
        num_joints = pos_hist.shape[1]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PID Control Results', fontsize=16, fontweight='bold')
        
        # Position tracking
        ax = axes[0, 0]
        for i in range(num_joints):
            ax.plot(time_hist, pos_hist[:, i], label=f'Joint {i+1}', linewidth=2)
            ax.axhline(y=targets[i], color=f'C{i}', linestyle='--', alpha=0.7)
        ax.set_title('Joint Positions', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (rad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Velocity profiles
        ax = axes[0, 1]
        for i in range(num_joints):
            ax.plot(time_hist, vel_hist[:, i], label=f'Joint {i+1}', linewidth=2)
        ax.set_title('Joint Velocities', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (rad/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Control torques
        ax = axes[1, 0]
        for i in range(num_joints):
            ax.plot(time_hist, torque_hist[:, i], label=f'Joint {i+1}', linewidth=2)
        ax.set_title('Control Torques', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Torque (Nm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Tracking errors
        ax = axes[1, 1]
        for i in range(num_joints):
            ax.plot(time_hist, error_hist[:, i], label=f'Joint {i+1}', linewidth=2)
        ax.set_title('Tracking Errors', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (rad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def analyze_step_response(self, time_hist, pos_hist, targets):
        """Analyze step response characteristics."""
        logger.info("\n📊 Step Response Analysis:")
        
        for joint in range(pos_hist.shape[1]):
            target = targets[joint]
            response = pos_hist[:, joint]
            
            # Rise time (10% to 90% of final value)
            final_value = response[-1]
            rise_start_idx = np.where(response >= 0.1 * target)[0]
            rise_end_idx = np.where(response >= 0.9 * target)[0]
            
            if len(rise_start_idx) > 0 and len(rise_end_idx) > 0:
                rise_time = time_hist[rise_end_idx[0]] - time_hist[rise_start_idx[0]]
            else:
                rise_time = float('inf')
            
            # Settling time (within 2% of target)
            settling_indices = np.where(np.abs(response - target) <= 0.02 * target)[0]
            if len(settling_indices) > 0:
                settling_time = time_hist[settling_indices[0]]
            else:
                settling_time = float('inf')
            
            # Overshoot
            max_response = np.max(response)
            overshoot = max(0, (max_response - target) / target * 100)
            
            # Steady-state error
            steady_state_error = abs(final_value - target)
            
            logger.info(f"  Joint {joint+1}:")
            logger.info(f"    Rise time: {rise_time:.3f}s")
            logger.info(f"    Settling time: {settling_time:.3f}s")
            logger.info(f"    Overshoot: {overshoot:.1f}%")
            logger.info(f"    Steady-state error: {steady_state_error:.4f} rad")
            
    def demonstrate_computed_torque_control(self):
        """Demonstrate computed torque control for trajectory tracking."""
        logger.info("\n🎯 Demonstrating Computed Torque Control...")
        
        # Generate a simple trajectory
        T_final = 5.0
        dt = 0.01
        time_steps = int(T_final / dt)
        num_joints = len(self.joint_limits)
        
        # Sinusoidal trajectory
        time_history = np.linspace(0, T_final, time_steps)
        amplitude = np.array([np.pi/6] * num_joints)
        frequency = np.array([0.5] * num_joints)
        
        # Desired trajectory
        desired_positions = np.array([
            amplitude * np.sin(2 * np.pi * frequency[i] * time_history)
            for i in range(num_joints)
        ]).T
        
        desired_velocities = np.array([
            amplitude * 2 * np.pi * frequency[i] * np.cos(2 * np.pi * frequency[i] * time_history)
            for i in range(num_joints)
        ]).T
        
        desired_accelerations = np.array([
            -amplitude * (2 * np.pi * frequency[i])**2 * np.sin(2 * np.pi * frequency[i] * time_history)
            for i in range(num_joints)
        ]).T
        
        # Control gains
        Kp = np.array([200.0] * num_joints)
        Ki = np.array([50.0] * num_joints)
        Kd = np.array([50.0] * num_joints)
        
        # Gravity vector
        gravity = np.array([0, 0, -9.81])
        Ftip = np.zeros(6)  # No external forces
        
        # Initial conditions
        current_positions = np.zeros(num_joints)
        current_velocities = np.zeros(num_joints)
        
        # Data storage
        actual_positions = []
        actual_velocities = []
        control_torques = []
        tracking_errors = []
        
        logger.info(f"Running computed torque simulation...")
        
        for step in range(time_steps):
            # Computed torque control
            torques = self.controller.computed_torque_control(
                thetalistd=desired_positions[step],
                dthetalistd=desired_velocities[step],
                ddthetalistd=desired_accelerations[step],
                thetalist=current_positions,
                dthetalist=current_velocities,
                g=gravity,
                dt=dt,
                Kp=Kp,
                Ki=Ki,
                Kd=Kd
            )
            
            # Convert from GPU if necessary
            if GPU_AVAILABLE and hasattr(torques, 'get'):
                torques = torques.get()
            
            # Apply torque limits
            torques = np.clip(torques, self.torque_limits[:, 0], self.torque_limits[:, 1])
            
            # Simplified dynamics integration
            mass_matrix = np.diag([1.0] * num_joints)
            accelerations = np.linalg.solve(mass_matrix, torques)
            
            # Update states
            current_velocities += accelerations * dt
            current_positions += current_velocities * dt
            
            # Store data
            actual_positions.append(current_positions.copy())
            actual_velocities.append(current_velocities.copy())
            control_torques.append(torques.copy())
            tracking_errors.append(desired_positions[step] - current_positions)
            
        # Convert to numpy arrays
        actual_positions = np.array(actual_positions)
        actual_velocities = np.array(actual_velocities)
        control_torques = np.array(control_torques)
        tracking_errors = np.array(tracking_errors)
        
        # Plot results
        self.plot_computed_torque_results(
            time_history, desired_positions, actual_positions,
            control_torques, tracking_errors
        )
        
        # Calculate RMS tracking error
        rms_errors = np.sqrt(np.mean(tracking_errors**2, axis=0))
        logger.info("📊 Tracking Performance:")
        for i, rms_error in enumerate(rms_errors):
            logger.info(f"  Joint {i+1} RMS error: {rms_error:.4f} rad")
            
        logger.info("✅ Computed torque control demonstration complete")
        
    def plot_computed_torque_results(self, time_hist, desired_pos, actual_pos, torques, errors):
        """Plot computed torque control results."""
        num_joints = actual_pos.shape[1]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Computed Torque Control Results', fontsize=16, fontweight='bold')
        
        # Position tracking
        ax = axes[0, 0]
        for i in range(num_joints):
            ax.plot(time_hist, desired_pos[:, i], '--', label=f'Desired {i+1}', 
                   linewidth=2, alpha=0.8)
            ax.plot(time_hist, actual_pos[:, i], '-', label=f'Actual {i+1}', 
                   linewidth=2)
        ax.set_title('Position Tracking', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (rad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Tracking errors
        ax = axes[0, 1]
        for i in range(num_joints):
            ax.plot(time_hist, errors[:, i], label=f'Joint {i+1}', linewidth=2)
        ax.set_title('Tracking Errors', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (rad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Control torques
        ax = axes[1, 0]
        for i in range(num_joints):
            ax.plot(time_hist, torques[:, i], label=f'Joint {i+1}', linewidth=2)
        ax.set_title('Control Torques', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Torque (Nm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Error statistics
        ax = axes[1, 1]
        rms_errors = np.sqrt(np.mean(errors**2, axis=0))
        max_errors = np.max(np.abs(errors), axis=0)
        
        x_pos = np.arange(num_joints)
        width = 0.35
        ax.bar(x_pos - width/2, rms_errors, width, label='RMS Error', alpha=0.8)
        ax.bar(x_pos + width/2, max_errors, width, label='Max Error', alpha=0.8)
        
        ax.set_title('Error Statistics', fontweight='bold')
        ax.set_xlabel('Joint Number')
        ax.set_ylabel('Error (rad)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'J{i+1}' for i in range(num_joints)])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def demonstrate_controller_tuning(self):
        """Demonstrate automatic controller tuning using Ziegler-Nichols method."""
        logger.info("\n🎯 Demonstrating Controller Tuning (Ziegler-Nichols)...")
        
        num_joints = len(self.joint_limits)
        
        # For demonstration, we'll use mock ultimate gain and period values
        # In practice, these would be determined experimentally
        ultimate_gains = np.array([150.0, 120.0, 100.0][:num_joints])
        ultimate_periods = np.array([0.8, 1.0, 1.2][:num_joints])
        
        logger.info("📐 Computing controller gains using Ziegler-Nichols method...")
        
        # Tune for different controller types
        controller_types = ['P', 'PI', 'PID']
        tuned_gains = {}
        
        for ctrl_type in controller_types:
            Kp, Ki, Kd = self.controller.ziegler_nichols_tuning(
                ultimate_gains, ultimate_periods, kind=ctrl_type
            )
            tuned_gains[ctrl_type] = {'Kp': Kp, 'Ki': Ki, 'Kd': Kd}
            
            logger.info(f"  {ctrl_type} Controller gains:")
            logger.info(f"    Kp: {Kp}")
            logger.info(f"    Ki: {Ki}")
            logger.info(f"    Kd: {Kd}")
            
        # Demonstrate the tuning method with actual ultimate gain finding
        logger.info("\n🔍 Finding ultimate gain and period for Joint 1...")
        
        # Use the controller's method to find ultimate parameters
        initial_angles = np.zeros(num_joints)
        target_angles = np.array([np.pi/6] * num_joints)
        
        try:
            ultimate_gain, ultimate_period, gain_history, error_history = \
                self.controller.find_ultimate_gain_and_period(
                    initial_angles, target_angles, dt=0.01, max_steps=500
                )
            
            logger.info(f"  Found ultimate gain: {ultimate_gain:.2f}")
            logger.info(f"  Found ultimate period: {ultimate_period:.3f}s")
            
            # Plot the gain search process
            self.plot_tuning_results(gain_history, error_history)
            
        except Exception as e:
            logger.warning(f"Ultimate gain finding failed: {e}")
            logger.info("Using predefined values for demonstration")
            
        logger.info("✅ Controller tuning demonstration complete")
        
    def plot_tuning_results(self, gain_history, error_history):
        """Plot controller tuning results."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('Controller Tuning Results', fontsize=16, fontweight='bold')
        
        # Gain progression
        ax = axes[0]
        ax.plot(gain_history, 'o-', linewidth=2, markersize=6)
        ax.set_title('Gain Progression During Tuning', fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Proportional Gain')
        ax.grid(True, alpha=0.3)
        
        # Error evolution for each gain
        ax = axes[1]
        for i, errors in enumerate(error_history[:min(5, len(error_history))]):
            if GPU_AVAILABLE and hasattr(errors, 'get'):
                errors = errors.get()
            time_steps = np.arange(len(errors)) * 0.01
            ax.plot(time_steps, errors, label=f'Gain {gain_history[i]:.1f}', 
                   linewidth=2, alpha=0.7)
        
        ax.set_title('Error Evolution for Different Gains', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('System Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def demonstrate_cartesian_control(self):
        """Demonstrate Cartesian space control."""
        logger.info("\n🎯 Demonstrating Cartesian Space Control...")
        
        num_joints = len(self.joint_limits)
        
        # Current joint configuration (safe starting position)
        current_joint_angles = self.get_safe_joint_targets() * 0.1  # Small initial offset
        current_joint_velocities = np.zeros(num_joints)
        
        # Desired Cartesian position (relative to current end-effector position)
        current_transform = self.robot.forward_kinematics(current_joint_angles)
        current_ee_pos = current_transform[:3, 3]
        
        # Move 10cm in positive X direction from current position
        desired_position = current_ee_pos + np.array([0.1, 0.0, 0.0])
        
        # Control gains
        Kp = np.array([100.0, 100.0, 100.0])  # Position gains
        Kd = np.array([20.0, 20.0, 20.0])     # Damping gains
        
        # Simulate control
        logger.info("Computing Cartesian control torques...")
        
        try:
            control_torques = self.controller.cartesian_space_control(
                desired_position=desired_position,
                current_joint_angles=current_joint_angles,
                current_joint_velocities=current_joint_velocities,
                Kp=Kp,
                Kd=Kd
            )
            
            # Convert from GPU if necessary
            if GPU_AVAILABLE and hasattr(control_torques, 'get'):
                control_torques = control_torques.get()
                
            # Current end-effector position
            current_transform = self.robot.forward_kinematics(current_joint_angles)
            current_position = current_transform[:3, 3]
            
            # Position error
            position_error = desired_position - current_position
            error_magnitude = np.linalg.norm(position_error)
            
            logger.info("📊 Cartesian Control Results:")
            logger.info(f"  Current EE position: [{current_position[0]:.3f}, {current_position[1]:.3f}, {current_position[2]:.3f}]")
            logger.info(f"  Desired EE position: [{desired_position[0]:.3f}, {desired_position[1]:.3f}, {desired_position[2]:.3f}]")
            logger.info(f"  Position error magnitude: {error_magnitude:.4f} m")
            logger.info(f"  Control torques: {control_torques}")
            
            # Visualize results
            self.plot_cartesian_control_results(
                current_position, desired_position, position_error, control_torques
            )
            
        except Exception as e:
            logger.error(f"Cartesian control failed: {e}")
            
        logger.info("✅ Cartesian space control demonstration complete")
        
    def plot_cartesian_control_results(self, current_pos, desired_pos, error, torques):
        """Plot Cartesian control results."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Cartesian Space Control Results', fontsize=16, fontweight='bold')
        
        # Position comparison
        ax = axes[0]
        coords = ['X', 'Y', 'Z']
        x_pos = np.arange(len(coords))
        width = 0.35
        
        ax.bar(x_pos - width/2, current_pos, width, label='Current', alpha=0.8)
        ax.bar(x_pos + width/2, desired_pos, width, label='Desired', alpha=0.8)
        
        ax.set_title('End-Effector Position', fontweight='bold')
        ax.set_xlabel('Coordinate')
        ax.set_ylabel('Position (m)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(coords)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Position error
        ax = axes[1]
        colors = ['red' if abs(e) > 0.01 else 'green' for e in error]
        bars = ax.bar(coords, error, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title('Position Error', fontweight='bold')
        ax.set_xlabel('Coordinate')
        ax.set_ylabel('Error (m)')
        ax.grid(True, alpha=0.3)
        
        # Add error values on bars
        for bar, err in zip(bars, error):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001*np.sign(height),
                   f'{err:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        # Control torques
        ax = axes[2]
        joint_labels = [f'J{i+1}' for i in range(len(torques))]
        colors = ['red' if abs(t) > 5.0 else 'blue' for t in torques]
        bars = ax.bar(joint_labels, torques, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title('Control Torques', fontweight='bold')
        ax.set_xlabel('Joint')
        ax.set_ylabel('Torque (Nm)')
        ax.grid(True, alpha=0.3)
        
        # Add torque values on bars
        for bar, torque in zip(bars, torques):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1*np.sign(height),
                   f'{torque:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.show()
        
    def demonstrate_feedforward_control(self):
        """Demonstrate PD control with feedforward compensation."""
        logger.info("\n🎯 Demonstrating PD + Feedforward Control...")
        
        num_joints = len(self.joint_limits)
        
        # Desired trajectory (safe targets for current robot)
        desired_position = self.get_safe_joint_targets() * 0.5  # Conservative targets
        desired_velocity = np.zeros(num_joints)
        desired_acceleration = np.zeros(num_joints)
        
        # Current state
        current_position = np.zeros(num_joints)
        current_velocity = np.zeros(num_joints)
        
        # Control gains
        Kp = np.array([150.0] * num_joints)
        Kd = np.array([30.0] * num_joints)
        
        # System parameters
        gravity = np.array([0, 0, -9.81])
        Ftip = np.zeros(6)
        
        logger.info("Computing feedforward control signals...")
        
        try:
            # Pure PD control
            pd_torques = self.controller.pd_control(
                desired_position=desired_position,
                desired_velocity=desired_velocity,
                current_position=current_position,
                current_velocity=current_velocity,
                Kp=Kp,
                Kd=Kd
            )
            
            # Pure feedforward control
            ff_torques = self.controller.feedforward_control(
                desired_position=desired_position,
                desired_velocity=desired_velocity,
                desired_acceleration=desired_acceleration,
                g=gravity,
                Ftip=Ftip
            )
            
            # Combined PD + Feedforward control
            combined_torques = self.controller.pd_feedforward_control(
                desired_position=desired_position,
                desired_velocity=desired_velocity,
                desired_acceleration=desired_acceleration,
                current_position=current_position,
                current_velocity=current_velocity,
                Kp=Kp,
                Kd=Kd,
                g=gravity,
                Ftip=Ftip
            )
            
            # Convert from GPU if necessary
            if GPU_AVAILABLE:
                if hasattr(pd_torques, 'get'):
                    pd_torques = pd_torques.get()
                if hasattr(ff_torques, 'get'):
                    ff_torques = ff_torques.get()
                if hasattr(combined_torques, 'get'):
                    combined_torques = combined_torques.get()
            
            logger.info("📊 Feedforward Control Analysis:")
            logger.info(f"  PD torques: {pd_torques}")
            logger.info(f"  Feedforward torques: {ff_torques}")
            logger.info(f"  Combined torques: {combined_torques}")
            
            # Analyze contribution of each component
            pd_contribution = np.abs(pd_torques) / (np.abs(combined_torques) + 1e-6) * 100
            ff_contribution = np.abs(ff_torques) / (np.abs(combined_torques) + 1e-6) * 100
            
            for i in range(num_joints):
                logger.info(f"  Joint {i+1}: PD={pd_contribution[i]:.1f}%, FF={ff_contribution[i]:.1f}%")
            
            # Plot results
            self.plot_feedforward_results(pd_torques, ff_torques, combined_torques)
            
        except Exception as e:
            logger.error(f"Feedforward control failed: {e}")
            
        logger.info("✅ PD + Feedforward control demonstration complete")
        
    def plot_feedforward_results(self, pd_torques, ff_torques, combined_torques):
        """Plot feedforward control analysis."""
        num_joints = len(pd_torques)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('PD + Feedforward Control Analysis', fontsize=16, fontweight='bold')
        
        # Torque comparison
        ax = axes[0]
        joint_labels = [f'J{i+1}' for i in range(num_joints)]
        x_pos = np.arange(num_joints)
        width = 0.25
        
        ax.bar(x_pos - width, pd_torques, width, label='PD Only', alpha=0.8)
        ax.bar(x_pos, ff_torques, width, label='Feedforward Only', alpha=0.8)
        ax.bar(x_pos + width, combined_torques, width, label='PD + Feedforward', alpha=0.8)
        
        ax.set_title('Control Torque Comparison', fontweight='bold')
        ax.set_xlabel('Joint')
        ax.set_ylabel('Torque (Nm)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(joint_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Component contribution
        ax = axes[1]
        pd_contribution = np.abs(pd_torques) / (np.abs(combined_torques) + 1e-6) * 100
        ff_contribution = np.abs(ff_torques) / (np.abs(combined_torques) + 1e-6) * 100
        
        ax.bar(x_pos, pd_contribution, width*2, label='PD Contribution', alpha=0.8)
        ax.bar(x_pos, ff_contribution, width*2, bottom=pd_contribution, 
               label='FF Contribution', alpha=0.8)
        
        ax.set_title('Control Component Contributions', fontweight='bold')
        ax.set_xlabel('Joint')
        ax.set_ylabel('Contribution (%)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(joint_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.show()
        
    def demonstrate_control_comparison(self):
        """Compare different control strategies on the same task."""
        logger.info("\n🎯 Demonstrating Control Strategy Comparison...")
        
        # Define a challenging trajectory
        T_final = 3.0
        dt = 0.01
        time_steps = int(T_final / dt)
        num_joints = len(self.joint_limits)
        
        time_history = np.linspace(0, T_final, time_steps)
        
        # Multi-frequency trajectory with safe amplitudes
        amplitude = self.get_trajectory_amplitudes() * 0.8  # Conservative amplitudes
        freq1 = np.array([0.3, 0.4, 0.5, 0.3, 0.4, 0.3, 0.5][:num_joints])
        freq2 = np.array([0.8, 0.6, 0.7, 0.9, 0.6, 0.8, 0.7][:num_joints])
        
        desired_positions = np.array([
            amplitude[i] * (np.sin(2*np.pi*freq1[i]*time_history) + 
                           0.3*np.sin(2*np.pi*freq2[i]*time_history))
            for i in range(num_joints)
        ]).T
        
        # Control strategies adapted for current robot
        if not self.use_simple_robot:  # XArm robot - more conservative gains
            strategies = {
                'PID': {'Kp': 50, 'Ki': 10, 'Kd': 15},
                'High_Gain_PD': {'Kp': 80, 'Ki': 0, 'Kd': 20},
                'Aggressive_PID': {'Kp': 100, 'Ki': 25, 'Kd': 30}
            }
        else:  # Simple robot - can handle higher gains
            strategies = {
                'PID': {'Kp': 100, 'Ki': 20, 'Kd': 25},
                'High_Gain_PD': {'Kp': 200, 'Ki': 0, 'Kd': 40},
                'Aggressive_PID': {'Kp': 300, 'Ki': 50, 'Kd': 60}
            }
        
        results = {}
        
        for strategy_name, gains in strategies.items():
            logger.info(f"  Testing {strategy_name} strategy...")
            
            # Reset initial conditions
            current_positions = np.zeros(num_joints)
            current_velocities = np.zeros(num_joints)
            
            # Reset controller state
            self.controller.eint = None
            
            positions_hist = []
            errors_hist = []
            torques_hist = []
            
            for step in range(min(500, time_steps)):  # Limit for demo
                if strategy_name in ['PID', 'Aggressive_PID']:
                    torques = self.controller.pid_control(
                        thetalistd=desired_positions[step],
                        dthetalistd=np.zeros(num_joints),
                        thetalist=current_positions,
                        dthetalist=current_velocities,
                        dt=dt,
                        Kp=np.array([gains['Kp']] * num_joints),
                        Ki=np.array([gains['Ki']] * num_joints),
                        Kd=np.array([gains['Kd']] * num_joints)
                    )
                else:  # PD control
                    torques = self.controller.pd_control(
                        desired_position=desired_positions[step],
                        desired_velocity=np.zeros(num_joints),
                        current_position=current_positions,
                        current_velocity=current_velocities,
                        Kp=np.array([gains['Kp']] * num_joints),
                        Kd=np.array([gains['Kd']] * num_joints)
                    )
                
                # Convert from GPU if necessary
                if GPU_AVAILABLE and hasattr(torques, 'get'):
                    torques = torques.get()
                
                # Apply limits and simple integration
                torques = np.clip(torques, self.torque_limits[:, 0], self.torque_limits[:, 1])
                accelerations = torques  # Simplified
                
                current_velocities += accelerations * dt * 0.1  # Damped
                current_positions += current_velocities * dt
                
                # Store results
                positions_hist.append(current_positions.copy())
                errors_hist.append(desired_positions[step] - current_positions)
                torques_hist.append(torques.copy())
            
            # Calculate performance metrics
            errors_array = np.array(errors_hist)
            rms_error = np.sqrt(np.mean(errors_array**2))
            max_error = np.max(np.abs(errors_array))
            
            results[strategy_name] = {
                'positions': np.array(positions_hist),
                'errors': errors_array,
                'torques': np.array(torques_hist),
                'rms_error': rms_error,
                'max_error': max_error,
                'time': time_history[:len(errors_hist)]
            }
            
            logger.info(f"    RMS Error: {rms_error:.4f} rad")
            logger.info(f"    Max Error: {max_error:.4f} rad")
        
        # Plot comparison
        self.plot_control_comparison(results, desired_positions)
        
        logger.info("✅ Control strategy comparison complete")
        
    def plot_control_comparison(self, results, desired_positions):
        """Plot control strategy comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Control Strategy Comparison', fontsize=16, fontweight='bold')
        
        # Position tracking for first joint
        ax = axes[0, 0]
        time_ref = results[list(results.keys())[0]]['time']
        ax.plot(time_ref, desired_positions[:len(time_ref), 0], 'k--', 
               linewidth=3, label='Desired', alpha=0.8)
        
        colors = ['blue', 'red', 'green']
        for i, (strategy, data) in enumerate(results.items()):
            ax.plot(data['time'], data['positions'][:, 0], 
                   color=colors[i], linewidth=2, label=strategy, alpha=0.8)
        
        ax.set_title('Joint 1 Position Tracking', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (rad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Tracking errors
        ax = axes[0, 1]
        for i, (strategy, data) in enumerate(results.items()):
            ax.plot(data['time'], data['errors'][:, 0], 
                   color=colors[i], linewidth=2, label=strategy, alpha=0.8)
        
        ax.set_title('Joint 1 Tracking Error', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (rad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Performance comparison
        ax = axes[1, 0]
        strategies = list(results.keys())
        rms_errors = [results[s]['rms_error'] for s in strategies]
        max_errors = [results[s]['max_error'] for s in strategies]
        
        x_pos = np.arange(len(strategies))
        width = 0.35
        
        ax.bar(x_pos - width/2, rms_errors, width, label='RMS Error', alpha=0.8)
        ax.bar(x_pos + width/2, max_errors, width, label='Max Error', alpha=0.8)
        
        ax.set_title('Performance Comparison', fontweight='bold')
        ax.set_xlabel('Control Strategy')
        ax.set_ylabel('Error (rad)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(strategies, rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Control effort comparison
        ax = axes[1, 1]
        avg_torques = [np.mean(np.abs(results[s]['torques'])) for s in strategies]
        max_torques = [np.max(np.abs(results[s]['torques'])) for s in strategies]
        
        ax.bar(x_pos - width/2, avg_torques, width, label='Avg Torque', alpha=0.8)
        ax.bar(x_pos + width/2, max_torques, width, label='Max Torque', alpha=0.8)
        
        ax.set_title('Control Effort Comparison', fontweight='bold')
        ax.set_xlabel('Control Strategy')
        ax.set_ylabel('Torque (Nm)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(strategies, rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def run_all_demonstrations(self):
        """Run all control demonstrations."""
        logger.info("🚀 Starting Basic Control Demonstrations")
        logger.info("=" * 60)
        
        try:
            # Run all demonstrations
            self.demonstrate_pid_control()
            self.demonstrate_computed_torque_control()
            self.demonstrate_controller_tuning()
            self.demonstrate_cartesian_control()
            self.demonstrate_feedforward_control()
            self.demonstrate_control_comparison()
            
            logger.info("\n" + "=" * 60)
            logger.info("🎉 All control demonstrations completed successfully!")
            
            # Summary
            logger.info("\n📋 Demonstration Summary:")
            logger.info("  ✅ PID Control - Step response and tuning")
            logger.info("  ✅ Computed Torque Control - Trajectory tracking")
            logger.info("  ✅ Ziegler-Nichols Tuning - Automatic gain calculation")
            logger.info("  ✅ Cartesian Space Control - End-effector positioning")
            logger.info("  ✅ Feedforward Control - Gravity compensation")
            logger.info("  ✅ Control Comparison - Performance analysis")
            
        except KeyboardInterrupt:
            logger.info("\n⏸️ Demonstrations interrupted by user")
        except Exception as e:
            logger.error(f"\n❌ Error during demonstrations: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function to run the basic control demo."""
    print("🤖 ManipulaPy Basic Control Demo - XArm Edition")
    print("=" * 60)
    
    # Check for GPU acceleration
    if GPU_AVAILABLE:
        print("🚀 GPU acceleration enabled")
    else:
        print("⚙️ Running on CPU only")
    
    print(f"📁 Using built-in XArm URDF: {urdf_file}")
    
    try:
        # Create and run the demo with XArm robot (default)
        demo = BasicControlDemo(use_simple_robot=False)
        
        # Print robot information
        num_joints = len(demo.joint_limits)
        robot_type = "XArm" if not demo.use_simple_robot else "Simple 3-DOF"
        print(f"🤖 Robot: {robot_type} ({num_joints} joints)")
        print(f"📐 Joint limits: {demo.joint_limits.shape}")
        print(f"⚡ Torque limits: {demo.torque_limits.shape}")
        
        demo.run_all_demonstrations()
        
        print("\n" + "=" * 60)
        print("🎯 Demo completed! Check the plots for detailed results.")
        print("💡 Try modifying control gains to see different behaviors.")
        print("🔧 The demo automatically adapts to the XArm robot's characteristics.")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try fallback to simple robot
        print("\n🔄 Attempting fallback to simple robot...")
        try:
            demo = BasicControlDemo(use_simple_robot=True)
            demo.run_all_demonstrations()
            print("✅ Fallback demo completed successfully!")
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            return 1
    
    return 0

if __name__ == "__main__":
    exit(main())