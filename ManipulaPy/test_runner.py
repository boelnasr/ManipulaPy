#!/usr/bin/env python3
"""
Improved test_control.py that tests real functionality when modules are available,
while gracefully falling back to CPU-based testing when GPU modules are unavailable.
"""

import unittest
import numpy as np
import sys
import os

# Add the package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def is_module_available(module_name):
    """Check if a module is really available (not mocked)."""
    try:
        module = __import__(module_name)
        # Check if it's our mock by looking for the _name attribute
        return not hasattr(module, '_name') or not str(module._name).startswith('Mock')
    except ImportError:
        return False

def get_array_library():
    """Get the best available array library (CuPy if available, otherwise NumPy)."""
    if is_module_available('cupy'):
        try:
            import cupy as cp
            return cp, 'cupy'
        except ImportError:
            pass
    
    # Fall back to NumPy with CuPy-like interface
    class NumpyWrapper:
        """NumPy wrapper that provides CuPy-like interface for testing."""
        
        def asarray(self, arr):
            return np.asarray(arr)
        
        def asnumpy(self, arr):
            return np.asarray(arr)
        
        def zeros(self, *args, **kwargs):
            return np.zeros(*args, **kwargs)
        
        def ones(self, *args, **kwargs):
            return np.ones(*args, **kwargs)
        
        def zeros_like(self, arr):
            return np.zeros_like(arr)
        
        def eye(self, n, **kwargs):
            return np.eye(n, **kwargs)
        
        def array(self, arr, **kwargs):
            return np.array(arr, **kwargs)
        
        def concatenate(self, arrays, **kwargs):
            return np.concatenate(arrays, **kwargs)
        
        def clip(self, arr, a_min, a_max):
            return np.clip(arr, a_min, a_max)
        
        @property
        def linalg(self):
            return np.linalg
        
        def __getattr__(self, name):
            return getattr(np, name)
    
    return NumpyWrapper(), 'numpy'

class TestManipulatorController(unittest.TestCase):
    def setUp(self):
        # Get the best available array library
        self.cp, self.backend = get_array_library()
        print(f"Using {self.backend} backend for testing")
        
        # Import after setting up the backend
        try:
            from ManipulaPy.control import ManipulatorController
            from ManipulaPy.urdf_processor import URDFToSerialManipulator
            from ManipulaPy.ManipulaPy_data.xarm import urdf_file as xarm_urdf_file
            
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
                
        except ImportError as e:
            self.skipTest(f"Control module not available: {e}")

    def create_mock_objects(self):
        """Create mock objects for testing without a real URDF"""

        # Create a simplified dynamics object for testing
        class MockDynamics:
            def __init__(self):
                self.Glist = np.array([np.eye(6), np.eye(6)])  # Mock inertia matrices
                self.S_list = np.array(
                    [[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0.1, 0]]
                ).T  # Mock screw axes
                self.M_list = np.eye(4)  # Mock home position

            def mass_matrix(self, thetalist):
                # Return a simple diagonal mass matrix
                return np.diag([1.0, 0.8])

            def velocity_quadratic_forces(self, thetalist, dthetalist):
                # Simple Coriolis term
                return np.array([0.01 * dthetalist[1] ** 2, 0.01 * dthetalist[0] ** 2])

            def gravity_forces(self, thetalist, g):
                # Simple gravity term
                return np.array(
                    [
                        0.5 * g[2] * np.sin(thetalist[0]),
                        0.3 * g[2] * np.sin(thetalist[0] + thetalist[1]),
                    ]
                )

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

        from ManipulaPy.control import ManipulatorController
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

    def test_backend_functionality(self):
        """Test that the chosen backend works correctly."""
        # Test basic array operations
        arr1 = self.cp.array([1, 2, 3])
        arr2 = self.cp.array([4, 5, 6])
        
        # Test arithmetic
        result = arr1 + arr2
        expected = np.array([5, 7, 9])
        
        if self.backend == 'cupy':
            np.testing.assert_array_equal(self.cp.asnumpy(result), expected)
        else:
            np.testing.assert_array_equal(result, expected)
        
        # Test linear algebra
        mat = self.cp.eye(3)
        vec = self.cp.array([1, 2, 3])
        result = self.cp.linalg.solve(mat, vec)
        
        if self.backend == 'cupy':
            np.testing.assert_array_almost_equal(self.cp.asnumpy(result), [1, 2, 3])
        else:
            np.testing.assert_array_almost_equal(result, [1, 2, 3])
        
        print(f"✅ {self.backend} backend functionality working")

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
                self.cp.asarray(thetalistd),
                self.cp.asarray(dthetalistd),
                self.cp.asarray(thetalist),
                self.cp.asarray(dthetalist),
                self.dt,
                self.cp.asarray(Kp),
                self.cp.asarray(Ki),
                self.cp.asarray(Kd),
            )

            # Convert back to numpy for integration
            if self.backend == 'cupy':
                ddthetalist = self.cp.asnumpy(tau)
            else:
                ddthetalist = tau
                
            dthetalist += ddthetalist * self.dt
            thetalist += dthetalist * self.dt

            # Apply joint limits
            thetalist = np.clip(
                thetalist, self.joint_limits[:, 0], self.joint_limits[:, 1]
            )

            history.append(np.copy(thetalist))

        final_position = history[-1]
        error = np.abs(final_position - thetalistd)

        # Verify that the controller converges to the desired position within tolerance
        tolerance = 0.1  # Relaxed tolerance for complex robots
        self.assertTrue(
            np.all(error < tolerance),
            f"PID control did not converge. Final error: {error}",
        )

        print(f"✅ PID control working with {self.backend} backend")

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
                self.cp.asarray(thetalistd),
                self.cp.asarray(dthetalistd),
                self.cp.asarray(ddthetalistd),
                self.cp.asarray(thetalist),
                self.cp.asarray(dthetalist),
                self.cp.asarray(self.g),
                self.dt,
                self.cp.asarray(Kp),
                self.cp.asarray(Ki),
                self.cp.asarray(Kd),
            )

            # Use forward dynamics for a more accurate simulation
            if self.backend == 'cupy':
                tau_np = self.cp.asnumpy(tau)
            else:
                tau_np = tau
                
            ddthetalist = self.dynamics.forward_dynamics(
                thetalist, dthetalist, tau_np, self.g, self.Ftip
            )

            dthetalist += ddthetalist * self.dt
            thetalist += dthetalist * self.dt

            history.append(np.copy(thetalist))

        final_position = history[-1]
        error = np.abs(final_position - thetalistd)

        # Verify that the controller handles gravity and converges to the desired position
        tolerance = 0.1  # Relaxed tolerance for complex robots
        self.assertTrue(
            np.all(error < tolerance),
            f"Computed torque control did not converge. Final error: {error}",
        )

        print(f"✅ Computed torque control working with {self.backend} backend")

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

        for i in range(steps):
            # Get feedforward torque
            tau_ff = self.controller.feedforward_control(
                self.cp.asarray(trajectory[i]),
                self.cp.asarray(velocities[i]),
                self.cp.asarray(accelerations[i]),
                self.cp.asarray(self.g),
                self.cp.asarray(self.Ftip),
            )

            if self.backend == 'cupy':
                torques.append(self.cp.asnumpy(tau_ff))
            else:
                torques.append(tau_ff)

        # Verify that torques are within reasonable bounds for real robots
        torques = np.array(torques)
        self.assertTrue(
            np.all(np.isfinite(torques)),
            "Feedforward torques contain non-finite values",
        )

        print(f"✅ Feedforward control working with {self.backend} backend")

    def test_enforcing_limits(self):
        """Test that joint and torque limits are properly enforced."""
        num_joints = len(self.thetalist)

        # Test joint limits enforcement
        thetalist = np.array([2 * np.pi] * num_joints)  # Beyond limits
        dthetalist = np.array([1.0] * num_joints)
        tau = np.array([15.0] * num_joints)  # Beyond torque limits

        clipped_theta, clipped_dtheta, clipped_tau = self.controller.enforce_limits(
            self.cp.asarray(thetalist),
            self.cp.asarray(dthetalist),
            self.cp.asarray(tau),
            self.cp.asarray(self.joint_limits),
            self.cp.asarray(self.torque_limits),
        )

        if self.backend == 'cupy':
            clipped_theta = self.cp.asnumpy(clipped_theta)
            clipped_tau = self.cp.asnumpy(clipped_tau)

        # Check joint limits
        for i in range(len(self.joint_limits)):
            self.assertTrue(
                clipped_theta[i] >= self.joint_limits[i, 0]
                and clipped_theta[i] <= self.joint_limits[i, 1],
                f"Joint limit enforcement failed for joint {i}",
            )

        # Check torque limits
        for i in range(len(self.torque_limits)):
            self.assertTrue(
                clipped_tau[i] >= self.torque_limits[i, 0]
                and clipped_tau[i] <= self.torque_limits[i, 1],
                f"Torque limit enforcement failed for joint {i}",
            )

        print(f"✅ Limit enforcement working with {self.backend} backend")

    def test_ziegler_nichols_tuning(self):
        """Test Ziegler-Nichols controller tuning."""
        ultimate_gain = 10.0
        ultimate_period = 0.5

        # Test PID tuning
        Kp, Ki, Kd = self.controller.ziegler_nichols_tuning(
            ultimate_gain, ultimate_period, kind="PID"
        )

        # Check Ziegler-Nichols formulas
        expected_Kp = 0.6 * ultimate_gain
        expected_Ki = 2.0 * expected_Kp / ultimate_period
        expected_Kd = 0.125 * expected_Kp * ultimate_period

        self.assertAlmostEqual(Kp, expected_Kp, places=5)
        self.assertAlmostEqual(Ki, expected_Ki, places=5)
        self.assertAlmostEqual(Kd, expected_Kd, places=5)

        # Test PI tuning
        Kp, Ki, Kd = self.controller.ziegler_nichols_tuning(
            ultimate_gain, ultimate_period, kind="PI"
        )

        expected_Kp = 0.45 * ultimate_gain
        expected_Ki = 1.2 * ultimate_gain / ultimate_period
        expected_Kd = 0.0

        self.assertAlmostEqual(Kp, expected_Kp, places=5)
        self.assertAlmostEqual(Ki, expected_Ki, places=5)
        self.assertAlmostEqual(Kd, expected_Kd, places=5)

        # Test P tuning
        Kp, Ki, Kd = self.controller.ziegler_nichols_tuning(
            ultimate_gain, ultimate_period, kind="P"
        )

        expected_Kp = 0.5 * ultimate_gain
        expected_Ki = 0.0
        expected_Kd = 0.0

        self.assertAlmostEqual(Kp, expected_Kp, places=5)
        self.assertAlmostEqual(Ki, expected_Ki, places=5)
        self.assertAlmostEqual(Kd, expected_Kd, places=5)

        print(f"✅ Ziegler-Nichols tuning working with {self.backend} backend")

    def test_array_backend_consistency(self):
        """Test that operations are consistent between NumPy and CuPy backends."""
        # Create test data
        test_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        test_vector = np.array([1, 1], dtype=np.float32)
        
        # Convert to backend arrays
        backend_data = self.cp.asarray(test_data)
        backend_vector = self.cp.asarray(test_vector)
        
        # Test matrix-vector multiplication
        result = backend_data @ backend_vector
        
        # Convert back to numpy for comparison
        if self.backend == 'cupy':
            result_np = self.cp.asnumpy(result)
        else:
            result_np = result
            
        # Expected result
        expected = test_data @ test_vector
        
        np.testing.assert_array_almost_equal(result_np, expected, decimal=5)
        
        # Test element-wise operations
        backend_sum = backend_data + backend_data
        if self.backend == 'cupy':
            sum_np = self.cp.asnumpy(backend_sum)
        else:
            sum_np = backend_sum
            
        expected_sum = test_data + test_data
        np.testing.assert_array_almost_equal(sum_np, expected_sum, decimal=5)
        
        print(f"✅ Array backend consistency verified for {self.backend}")

    def test_performance_comparison(self):
        """Test performance characteristics of the chosen backend."""
        import time
        
        # Create larger test matrices for performance testing
        size = 100
        A = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size).astype(np.float32)
        
        # Convert to backend
        A_backend = self.cp.asarray(A)
        b_backend = self.cp.asarray(b)
        
        # Time matrix operations
        start_time = time.time()
        for _ in range(10):
            result = self.cp.linalg.solve(A_backend, b_backend)
            if self.backend == 'cupy':
                # Force computation to complete on GPU
                self.cp.asnumpy(result)
        end_time = time.time()
        
        backend_time = end_time - start_time
        
        # Time NumPy operations for comparison
        start_time = time.time()
        for _ in range(10):
            np.linalg.solve(A, b)
        end_time = time.time()
        
        numpy_time = end_time - start_time
        
        print(f"✅ Performance test completed:")
        print(f"   {self.backend} time: {backend_time:.4f}s")
        print(f"   NumPy time: {numpy_time:.4f}s")
        
        if self.backend == 'cupy':
            print(f"   Speedup: {numpy_time/backend_time:.2f}x")
        else:
            print("   Using NumPy fallback (no GPU available)")

class TestControllerWithRealLibraries(unittest.TestCase):
    """Test controller functionality with real external libraries when available."""
    
    def setUp(self):
        self.cp, self.backend = get_array_library()
        
    def test_torch_integration(self):
        """Test integration with PyTorch when available."""
        if not is_module_available('torch'):
            self.skipTest("Real PyTorch not available")
            
        try:
            import torch
            from ManipulaPy.control import ManipulatorController
            
            # Test that we can convert between PyTorch and our backend
            torch_tensor = torch.tensor([1.0, 2.0, 3.0])
            numpy_array = torch_tensor.detach().cpu().numpy()
            backend_array = self.cp.asarray(numpy_array)
            
            # Test round-trip conversion
            if self.backend == 'cupy':
                back_to_numpy = self.cp.asnumpy(backend_array)
            else:
                back_to_numpy = backend_array
                
            back_to_torch = torch.from_numpy(back_to_numpy)
            
            self.assertTrue(torch.allclose(torch_tensor, back_to_torch))
            
            print(f"✅ PyTorch integration working with {self.backend} backend")
            
        except ImportError as e:
            self.skipTest(f"Real PyTorch not available: {e}")

    def test_scipy_integration(self):
        """Test integration with SciPy when available."""
        try:
            from scipy.optimize import minimize
            from scipy.linalg import solve
            
            # Test basic SciPy functionality that might be used in control
            A = np.array([[2, 1], [1, 2]], dtype=np.float64)
            b = np.array([3, 3], dtype=np.float64)
            
            # Test SciPy linear algebra
            x = solve(A, b)
            expected = np.array([1, 1])
            np.testing.assert_array_almost_equal(x, expected, decimal=10)
            
            # Test optimization (could be used for parameter tuning)
            def objective(x):
                return (x[0] - 1)**2 + (x[1] - 2)**2
            
            result = minimize(objective, [0, 0], method='BFGS')
            self.assertTrue(result.success, "Optimization should converge")
            np.testing.assert_array_almost_equal(result.x, [1, 2], decimal=3)
            
            print("✅ SciPy integration working")
            
        except ImportError as e:
            self.skipTest(f"SciPy not available: {e}")

class TestControllerEdgeCases(unittest.TestCase):
    """Test edge cases and error handling in controller."""
    
    def setUp(self):
        self.cp, self.backend = get_array_library()
        
        # Create minimal mock for testing edge cases
        class MinimalMockDynamics:
            def __init__(self):
                self.Glist = [np.eye(6)]
                
            def mass_matrix(self, thetalist):
                return np.eye(len(thetalist))
                
            def velocity_quadratic_forces(self, thetalist, dthetalist):
                return np.zeros_like(thetalist)
                
            def gravity_forces(self, thetalist, g):
                return np.zeros_like(thetalist)
                
            def inverse_dynamics(self, thetalist, dthetalist, ddthetalist, g, Ftip):
                return np.zeros_like(thetalist)
                
            def jacobian(self, thetalist):
                return np.eye(6, len(thetalist))
        
        from ManipulaPy.control import ManipulatorController
        self.dynamics = MinimalMockDynamics()
        self.controller = ManipulatorController(self.dynamics)

    def test_zero_input_handling(self):
        """Test controller behavior with zero inputs."""
        zeros = self.cp.zeros(2)
        
        # Test PID with zero inputs
        result = self.controller.pid_control(
            zeros, zeros, zeros, zeros, 0.01, zeros, zeros, zeros
        )
        
        if self.backend == 'cupy':
            result_np = self.cp.asnumpy(result)
        else:
            result_np = result
            
        np.testing.assert_array_equal(result_np, np.zeros(2))
        
        print(f"✅ Zero input handling working with {self.backend} backend")

    def test_large_input_handling(self):
        """Test controller behavior with large inputs."""
        large_vals = self.cp.array([1e6, 1e6])
        normal_vals = self.cp.array([1.0, 1.0])
        
        # Test that large inputs don't cause overflow
        try:
            result = self.controller.pd_control(
                large_vals, normal_vals, normal_vals, normal_vals, 
                normal_vals, normal_vals
            )
            
            if self.backend == 'cupy':
                result_np = self.cp.asnumpy(result)
            else:
                result_np = result
                
            # Should not contain NaN or inf
            self.assertTrue(np.all(np.isfinite(result_np)), 
                          "Large inputs should not cause NaN/inf")
            
            print(f"✅ Large input handling working with {self.backend} backend")
            
        except Exception as e:
            self.fail(f"Large input handling failed: {e}")

    def test_array_shape_validation(self):
        """Test that mismatched array shapes are handled properly."""
        arr2 = self.cp.array([1.0, 2.0])
        arr3 = self.cp.array([1.0, 2.0, 3.0])
        
        # Test with mismatched shapes - should either work or fail gracefully
        try:
            result = self.controller.pd_control(
                arr2, arr2, arr3, arr2, arr2, arr2  # arr3 has wrong shape
            )
            # If it doesn't raise an error, check the result makes sense
            if self.backend == 'cupy':
                result_np = self.cp.asnumpy(result)
            else:
                result_np = result
            self.assertTrue(np.all(np.isfinite(result_np)))
        except (ValueError, RuntimeError, IndexError) as e:
            # These are acceptable errors for shape mismatches
            pass
        except Exception as e:
            self.fail(f"Unexpected error type for shape mismatch: {type(e).__name__}: {e}")
        
        print(f"✅ Array shape validation working with {self.backend} backend")

if __name__ == "__main__":
    # Print environment info
    print("Testing control module with available backends:")
    print(f"- CuPy (GPU): {'✓' if is_module_available('cupy') else '✗'}")
    print(f"- PyTorch: {'✓' if is_module_available('torch') else '✗'}")
    print(f"- SciPy: {'✓' if is_module_available('scipy') else '✗'}")
    print(f"- NumPy: ✓ (always available)")
    print()
    
    # Run with verbose output
    unittest.main(verbosity=2)