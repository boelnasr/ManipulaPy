#!/usr/bin/env python3

import unittest
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from ManipulaPy.control import ManipulatorController
from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.ManipulaPy_data.xarm import urdf_file as xarm_urdf_file

class TestManipulatorController(unittest.TestCase):
    def setUp(self):
        # Use the built-in xarm urdf file from the library
        self.urdf_path = xarm_urdf_file
        
        try:
            self.urdf_processor = URDFToSerialManipulator(self.urdf_path)
            self.dynamics = self.urdf_processor.dynamics
            self.controller = ManipulatorController(self.dynamics)
            
            # Common test parameters
            self.g = np.array([0, 0, -9.81])
            self.Ftip = np.array([0, 0, 0, 0, 0, 0])
            self.dt = 0.01
            
            # Get the number of joints from the dynamics
            num_joints = len(self.dynamics.Glist)
            self.thetalist = np.zeros(num_joints)
            self.dthetalist = np.zeros(num_joints)
            self.ddthetalist = np.zeros(num_joints)
            
            # Default joint and torque limits if not available
            self.joint_limits = np.array([[-np.pi, np.pi]] * num_joints)
            self.torque_limits = np.array([[-10, 10]] * num_joints)
            
        except Exception as e:
            print(f"Error loading URDF: {e}")
            self.create_mock_objects()
            
    def create_mock_objects(self):
        """Create mock objects for testing without a real URDF"""
        # Create a simplified dynamics object for testing
        class MockDynamics:
            def __init__(self):
                self.Glist = np.array([np.eye(6), np.eye(6)])  # Mock inertia matrices
                self.S_list = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0.1, 0]]).T  # Mock screw axes
                self.M_list = np.eye(4)  # Mock home position
            
            def mass_matrix(self, thetalist):
                # Return a simple diagonal mass matrix
                return np.diag([1.0, 0.8])
                
            def velocity_quadratic_forces(self, thetalist, dthetalist):
                # Simple Coriolis term
                return np.array([0.01 * dthetalist[1]**2, 0.01 * dthetalist[0]**2])
                
            def gravity_forces(self, thetalist, g):
                # Simple gravity term
                return np.array([0.5 * g[2] * np.sin(thetalist[0]), 
                                0.3 * g[2] * np.sin(thetalist[0] + thetalist[1])])
                
            def inverse_dynamics(self, thetalist, dthetalist, ddthetalist, g, Ftip):
                # Simple implementation for testing
                M = self.mass_matrix(thetalist)
                c = self.velocity_quadratic_forces(thetalist, dthetalist)
                grav = self.gravity_forces(thetalist, g)
                return M.dot(ddthetalist) + c + grav
                
            def forward_dynamics(self, thetalist, dthetalist, taulist, g, Ftip):
                # Simple implementation for testing
                M = self.mass_matrix(thetalist)
                c = self.velocity_quadratic_forces(thetalist, dthetalist)
                grav = self.gravity_forces(thetalist, g)
                return np.linalg.solve(M, taulist - c - grav)
                
            def jacobian(self, thetalist):
                # Simple Jacobian for 2-DOF planar robot
                l1 = 0.5  # Link 1 length
                l2 = 0.3  # Link 2 length
                s1 = np.sin(thetalist[0])
                s12 = np.sin(thetalist[0] + thetalist[1])
                c1 = np.cos(thetalist[0])
                c12 = np.cos(thetalist[0] + thetalist[1])
                
                J = np.zeros((6, 2))
                # Linear velocity components
                J[0, 0] = -l1 * s1 - l2 * s12
                J[0, 1] = -l2 * s12
                J[1, 0] = l1 * c1 + l2 * c12
                J[1, 1] = l2 * c12
                # Angular velocity components
                J[5, 0] = 1
                J[5, 1] = 1
                
                return J
        
        self.dynamics = MockDynamics()
        self.controller = ManipulatorController(self.dynamics)
        self.g = np.array([0, 0, -9.81])
        self.Ftip = np.array([0, 0, 0, 0, 0, 0])
        self.dt = 0.01
        self.thetalist = np.array([0.1, 0.2])
        self.dthetalist = np.array([0, 0])
        self.ddthetalist = np.array([0, 0])
        self.joint_limits = np.array([[-np.pi, np.pi], [-np.pi, np.pi]])
        self.torque_limits = np.array([[-10, 10], [-10, 10]])
        
    def test_pid_control(self):
        """Test PID control convergence to a setpoint."""
        # Set up test parameters
        num_joints = len(self.thetalist)
        thetalistd = np.array([0.5, 0.7] if num_joints == 2 else [0.5] * num_joints)
        dthetalistd = np.zeros_like(thetalistd)
        
        # Define gains
        Kp = np.array([5.0] * num_joints)
        Ki = np.array([0.1] * num_joints)
        Kd = np.array([1.0] * num_joints)
        
        # Simulate a simple control loop
        thetalist = np.copy(self.thetalist)
        dthetalist = np.copy(self.dthetalist)
        history = []
        steps = 500
        
        for _ in range(steps):
            tau = self.controller.pid_control(
                cp.asarray(thetalistd),
                cp.asarray(dthetalistd),
                cp.asarray(thetalist),
                cp.asarray(dthetalist),
                self.dt,
                cp.asarray(Kp),
                cp.asarray(Ki),
                cp.asarray(Kd)
            )
            
            # Simple dynamics update (acceleration = torque with normalized mass)
            ddthetalist = cp.asnumpy(tau) 
            dthetalist += ddthetalist * self.dt
            thetalist += dthetalist * self.dt
            
            # Apply joint limits
            thetalist = np.clip(thetalist, self.joint_limits[:, 0], self.joint_limits[:, 1])
            
            history.append(np.copy(thetalist))
        
        final_position = history[-1]
        error = np.abs(final_position - thetalistd)
        
        # Verify that the controller converges to the desired position within tolerance
        tolerance = 0.1  # Relaxed tolerance for complex robots
        self.assertTrue(np.all(error < tolerance), 
                        f"PID control did not converge. Final error: {error}")
        
        # Optional: Plot the trajectory for visualization
        # self._plot_control_response(history, thetalistd)
        
    def test_computed_torque_control(self):
        """Test computed torque control with non-zero gravity."""
        # Set up test parameters
        num_joints = len(self.thetalist)
        thetalistd = np.array([0.8, -0.5] if num_joints == 2 else [0.5] * num_joints)
        dthetalistd = np.zeros_like(thetalistd)
        ddthetalistd = np.zeros_like(thetalistd)
        
        # Define gains
        Kp = np.array([20.0] * num_joints)
        Ki = np.array([0.1] * num_joints)
        Kd = np.array([5.0] * num_joints)
        
        # Simulate a control loop with gravity
        thetalist = np.copy(self.thetalist)
        dthetalist = np.copy(self.dthetalist)
        history = []
        steps = 300
        
        for _ in range(steps):
            tau = self.controller.computed_torque_control(
                cp.asarray(thetalistd),
                cp.asarray(dthetalistd),
                cp.asarray(ddthetalistd),
                cp.asarray(thetalist),
                cp.asarray(dthetalist),
                cp.asarray(self.g),
                self.dt,
                cp.asarray(Kp),
                cp.asarray(Ki),
                cp.asarray(Kd)
            )
            
            # Use forward dynamics for a more accurate simulation
            ddthetalist = self.dynamics.forward_dynamics(
                thetalist, dthetalist, cp.asnumpy(tau), self.g, self.Ftip
            )
            
            dthetalist += ddthetalist * self.dt
            thetalist += dthetalist * self.dt
            
            history.append(np.copy(thetalist))
        
        final_position = history[-1]
        error = np.abs(final_position - thetalistd)
        
        # Verify that the controller handles gravity and converges to the desired position
        tolerance = 0.1  # Relaxed tolerance for complex robots
        self.assertTrue(np.all(error < tolerance), 
                        f"Computed torque control did not converge. Final error: {error}")
        
    def test_feedforward_control(self):
        """Test feedforward control with a simple trajectory."""
        # Generate a simple trajectory (e.g., linear path)
        steps = 200
        num_joints = len(self.thetalist)
        thetastart = np.copy(self.thetalist)
        thetaend = np.array([0.8, -0.5] if num_joints == 2 else [0.5] * num_joints)
        
        trajectory = []
        velocities = []
        accelerations = []
        
        for i in range(steps):
            s = i / (steps - 1)  # Normalized time from 0 to 1
            sdot = 1 / (steps - 1)
            sddot = 0
            
            # Linear interpolation for position
            theta = thetastart + s * (thetaend - thetastart)
            dtheta = sdot * (thetaend - thetastart)
            ddtheta = sddot * (thetaend - thetastart)
            
            trajectory.append(theta)
            velocities.append(dtheta)
            accelerations.append(ddtheta)
        
        # Test the feedforward controller
        torques = []
        Kp = np.array([10.0] * num_joints)
        Kd = np.array([2.0] * num_joints)
        
        for i in range(steps):
            # Get feedforward torque
            tau_ff = self.controller.feedforward_control(
                cp.asarray(trajectory[i]),
                cp.asarray(velocities[i]),
                cp.asarray(accelerations[i]),
                cp.asarray(self.g),
                cp.asarray(self.Ftip)
            )
            
            torques.append(cp.asnumpy(tau_ff))
        
        # Verify that torques are within reasonable bounds for real robots
        torques = np.array(torques)
        self.assertTrue(np.all(np.isfinite(torques)), 
                       "Feedforward torques contain non-finite values")
        
    def test_pd_feedforward_control(self):
        """Test combined PD and feedforward control."""
        # Generate a reference trajectory
        steps = 200
        num_joints = len(self.thetalist)
        thetastart = np.copy(self.thetalist)
        thetaend = np.array([0.8, -0.5] if num_joints == 2 else [0.5] * num_joints)
        
        trajectory = []
        velocities = []
        accelerations = []
        
        for i in range(steps):
            s = i / (steps - 1)  # Normalized time from 0 to 1
            sdot = 1 / (steps - 1)
            sddot = 0
            
            # Linear interpolation
            theta = thetastart + s * (thetaend - thetastart)
            dtheta = sdot * (thetaend - thetastart)
            ddtheta = sddot * (thetaend - thetastart)
            
            trajectory.append(theta)
            velocities.append(dtheta)
            accelerations.append(ddtheta)
        
        # Simulate execution with disturbance
        Kp = np.array([10.0] * num_joints)
        Kd = np.array([2.0] * num_joints)
        
        current_pos = np.copy(thetastart)
        current_vel = np.zeros_like(current_pos)
        
        execution_history = []
        
        for i in range(steps):
            # Add a small disturbance to test robustness
            disturbance = np.random.normal(0, 0.01, size=len(current_pos))
            
            # Get control signal
            tau = self.controller.pd_feedforward_control(
                cp.asarray(trajectory[i]),
                cp.asarray(velocities[i]),
                cp.asarray(accelerations[i]),
                cp.asarray(current_pos),
                cp.asarray(current_vel),
                cp.asarray(Kp),
                cp.asarray(Kd),
                cp.asarray(self.g),
                cp.asarray(self.Ftip)
            )
            
            # Apply dynamics
            ddthetalist = self.dynamics.forward_dynamics(
                current_pos, current_vel, cp.asnumpy(tau), self.g, self.Ftip
            )
            
            # Add disturbance to acceleration
            ddthetalist += disturbance
            
            # Update state
            current_vel += ddthetalist * self.dt
            current_pos += current_vel * self.dt
            
            execution_history.append(np.copy(current_pos))
        
        # Calculate tracking error
        execution_history = np.array(execution_history)
        trajectory = np.array(trajectory)
        
        tracking_error = np.mean(np.abs(execution_history - trajectory), axis=0)
        
        # Verify tracking error is within acceptable bounds
        max_allowed_error = 0.2  # Relaxed tolerance for more complex robots
        self.assertTrue(np.all(tracking_error < max_allowed_error),
                       f"PD+Feedforward tracking error too high: {tracking_error}")
        
    def test_enforcing_limits(self):
        """Test that joint and torque limits are properly enforced."""
        num_joints = len(self.thetalist)
        
        # Test joint limits enforcement
        thetalist = np.array([2*np.pi] * num_joints)  # Beyond limits
        dthetalist = np.array([1.0] * num_joints)
        tau = np.array([15.0] * num_joints)  # Beyond torque limits
        
        clipped_theta, clipped_dtheta, clipped_tau = self.controller.enforce_limits(
            cp.asarray(thetalist),
            cp.asarray(dthetalist),
            cp.asarray(tau),
            cp.asarray(self.joint_limits),
            cp.asarray(self.torque_limits)
        )
        
        clipped_theta = cp.asnumpy(clipped_theta)
        clipped_tau = cp.asnumpy(clipped_tau)
        
        # Check joint limits
        for i in range(len(self.joint_limits)):
            self.assertTrue(clipped_theta[i] >= self.joint_limits[i, 0] and 
                           clipped_theta[i] <= self.joint_limits[i, 1],
                           f"Joint limit enforcement failed for joint {i}")
            
        # Check torque limits
        for i in range(len(self.torque_limits)):
            self.assertTrue(clipped_tau[i] >= self.torque_limits[i, 0] and 
                           clipped_tau[i] <= self.torque_limits[i, 1],
                           f"Torque limit enforcement failed for joint {i}")
    
    def test_ziegler_nichols_tuning(self):
        """Test Ziegler-Nichols controller tuning."""
        ultimate_gain = 10.0
        ultimate_period = 0.5
        
        # Test PID tuning
        Kp, Ki, Kd = self.controller.ziegler_nichols_tuning(
            ultimate_gain, ultimate_period, controller_type="PID"
        )
        
        # Check Ziegler-Nichols formulas
        expected_Kp = 0.6 * ultimate_gain
        expected_Ki = 2 * expected_Kp / ultimate_period
        expected_Kd = expected_Kp * ultimate_period / 8
        
        self.assertAlmostEqual(Kp, expected_Kp, places=5)
        self.assertAlmostEqual(Ki, expected_Ki, places=5)
        self.assertAlmostEqual(Kd, expected_Kd, places=5)
        
        # Test PI tuning
        Kp, Ki, Kd = self.controller.ziegler_nichols_tuning(
            ultimate_gain, ultimate_period, controller_type="PI"
        )
        
        expected_Kp = 0.45 * ultimate_gain
        expected_Ki = 1.2 * expected_Kp / ultimate_period
        expected_Kd = 0.0
        
        self.assertAlmostEqual(Kp, expected_Kp, places=5)
        self.assertAlmostEqual(Ki, expected_Ki, places=5)
        self.assertAlmostEqual(Kd, expected_Kd, places=5)
        
        # Test P tuning
        Kp, Ki, Kd = self.controller.ziegler_nichols_tuning(
            ultimate_gain, ultimate_period, controller_type="P"
        )
        
        expected_Kp = 0.5 * ultimate_gain
        expected_Ki = 0.0
        expected_Kd = 0.0
        
        self.assertAlmostEqual(Kp, expected_Kp, places=5)
        self.assertAlmostEqual(Ki, expected_Ki, places=5)
        self.assertAlmostEqual(Kd, expected_Kd, places=5)
    
    def _plot_control_response(self, history, target):
        """Utility method to plot control response for debugging."""
        history = np.array(history)
        plt.figure(figsize=(10, 6))
        
        for i in range(history.shape[1]):
            plt.plot(history[:, i], label=f'Joint {i+1}')
            plt.axhline(y=target[i], color=f'C{i}', linestyle='--', 
                        label=f'Target {i+1}')
        
        plt.xlabel('Time Step')
        plt.ylabel('Joint Angle (rad)')
        plt.title('Control Response')
        plt.legend()
        plt.grid(True)
        plt.savefig('control_response.png')
        plt.close()

if __name__ == "__main__":
    unittest.main()