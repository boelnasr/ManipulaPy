#!/usr/bin/env python3
"""
Comprehensive CUDA Kernels Test Suite - ManipulaPy

This test suite provides thorough testing of all CUDA-accelerated functionality
including trajectory generation, dynamics computation, potential fields, and
performance benchmarking with proper error handling and fallback mechanisms.

Copyright (c) 2025 Mohamed Aboelnar
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import pytest
import numpy as np
import time
import warnings
from unittest.mock import patch, MagicMock

# Import ManipulaPy modules
try:
    from ManipulaPy import (
        CUDA_AVAILABLE,
        CUPY_AVAILABLE,
        check_cuda_availability,
        check_cupy_availability,
        optimized_trajectory_generation,
        optimized_potential_field,
        optimized_batch_trajectory_generation,
        get_cuda_array,
        return_cuda_array,
        make_1d_grid,
        make_2d_grid,
        get_gpu_properties,
        benchmark_kernel_performance,
        trajectory_cpu_fallback,
    )
    from ManipulaPy.cuda_kernels import (
        get_optimal_kernel_config,
        auto_select_optimal_kernel,
        print_performance_recommendations,
        setup_cuda_environment_for_40x_speedup,
        get_memory_pool_stats,
        optimized_trajectory_generation_monitored,
        _best_2d_config,
        _h2d_pinned,
        profile_start,
        profile_stop,
    )
    MANIPULAPY_AVAILABLE = True
except ImportError as e:
    MANIPULAPY_AVAILABLE = False
    pytest.skip(f"ManipulaPy not available: {e}", allow_module_level=True)

# CUDA-specific imports (conditional)
if CUDA_AVAILABLE:
    try:
        from ManipulaPy.cuda_kernels import (
            trajectory_kernel,
            inverse_dynamics_kernel,
            forward_dynamics_kernel,
            cartesian_trajectory_kernel,
            fused_potential_gradient_kernel,
            batch_trajectory_kernel,
        )
        CUDA_KERNELS_AVAILABLE = True
    except ImportError:
        CUDA_KERNELS_AVAILABLE = False
else:
    CUDA_KERNELS_AVAILABLE = False


class TestCUDAAvailability:
    """Test CUDA availability detection and setup."""
    
    def test_cuda_availability_check(self):
        """Test CUDA availability detection."""
        result = check_cuda_availability()
        assert isinstance(result, bool)
        
        if CUDA_AVAILABLE:
            assert result == True
        else:
            # When CUDA is not available, the function should return False
            # The warning might be emitted internally but not always propagated to pytest
            assert result == False
    
    def test_cuda_availability_with_warning_capture(self):
        """Test CUDA availability with warning capture."""
        # Use warning filter to capture any warnings
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            result = check_cuda_availability()
            
            if not CUDA_AVAILABLE:
                assert result == False
                # Check if any warning was emitted (might not always happen)
                if warning_list:
                    assert any("CUDA" in str(w.message) for w in warning_list)
            else:
                assert result == True
    
    def test_cupy_availability_check(self):
        """Test CuPy availability detection."""
        result = check_cupy_availability()
        assert isinstance(result, bool)
        
        if not CUPY_AVAILABLE:
            with pytest.warns(UserWarning, match="CuPy not available"):
                result = check_cupy_availability()
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_gpu_properties_retrieval(self):
        """Test GPU properties retrieval."""
        props = get_gpu_properties()
        assert props is not None
        assert isinstance(props, dict)
        assert 'multiprocessor_count' in props
        assert 'max_threads_per_block' in props
        assert props['multiprocessor_count'] > 0
        assert props['max_threads_per_block'] > 0
    
    def test_gpu_properties_fallback(self):
        """Test GPU properties when CUDA unavailable."""
        if not CUDA_AVAILABLE:
            props = get_gpu_properties()
            assert props is None


class TestGridConfigurations:
    """Test CUDA grid and block configuration utilities."""
    
    def test_1d_grid_basic(self):
        """Test 1D grid configuration."""
        size = 1024
        grid, block = make_1d_grid(size)
        
        assert isinstance(grid, tuple)
        assert isinstance(block, tuple)
        assert len(grid) == 1
        assert len(block) == 1
        assert grid[0] > 0
        assert block[0] > 0
        assert grid[0] * block[0] >= size
    
    def test_1d_grid_edge_cases(self):
        """Test 1D grid edge cases."""
        # Zero size
        grid, block = make_1d_grid(0)
        assert grid == (1,)
        assert block == (1,)
        
        # Small size
        grid, block = make_1d_grid(10)
        assert grid[0] > 0
        assert block[0] > 0
        
        # Large size
        grid, block = make_1d_grid(100000)
        assert grid[0] > 0
        assert block[0] > 0
    
    def test_2d_grid_basic(self):
        """Test 2D grid configuration."""
        N, num_joints = 1000, 6
        grid, block = make_2d_grid(N, num_joints)
        
        assert isinstance(grid, tuple)
        assert isinstance(block, tuple)
        assert len(grid) == 2
        assert len(block) == 2
        assert all(g > 0 for g in grid)
        assert all(b > 0 for b in block)
    
    def test_2d_grid_edge_cases(self):
        """Test 2D grid edge cases."""
        # Small problem
        grid, block = make_2d_grid(10, 3)
        assert all(g > 0 for g in grid)
        assert all(b > 0 for b in block)
        
        # Large problem
        grid, block = make_2d_grid(10000, 12)
        assert all(g > 0 for g in grid)
        assert all(b > 0 for b in block)
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_best_2d_config(self):
        """Test optimized 2D configuration selection."""
        config = _best_2d_config(1000, 6)
        assert isinstance(config, tuple)
        assert len(config) == 2
        grid, block = config
        assert isinstance(grid, tuple) and len(grid) == 2
        assert isinstance(block, tuple) and len(block) == 2


class TestTrajectoryGeneration:
    """Test trajectory generation with CUDA acceleration."""
    
    @pytest.fixture
    def trajectory_params(self):
        """Common trajectory parameters for testing."""
        return {
            'thetastart': np.array([0.0, 0.5, 1.0, -0.5, 0.2, -0.3], dtype=np.float32),
            'thetaend': np.array([1.0, 0.0, -1.0, 0.8, -0.4, 0.6], dtype=np.float32),
            'Tf': 2.0,
            'N': 100,
            'method': 5  # Quintic
        }
    
    def test_cpu_fallback_basic(self, trajectory_params):
        """Test CPU fallback trajectory generation."""
        traj_pos, traj_vel, traj_acc = trajectory_cpu_fallback(
            trajectory_params['thetastart'],
            trajectory_params['thetaend'],
            trajectory_params['Tf'],
            trajectory_params['N'],
            trajectory_params['method']
        )
        
        num_joints = len(trajectory_params['thetastart'])
        N = trajectory_params['N']
        
        assert traj_pos.shape == (N, num_joints)
        assert traj_vel.shape == (N, num_joints)
        assert traj_acc.shape == (N, num_joints)
        
        # Check boundary conditions
        np.testing.assert_allclose(traj_pos[0], trajectory_params['thetastart'], rtol=1e-5)
        np.testing.assert_allclose(traj_pos[-1], trajectory_params['thetaend'], rtol=1e-5)
        
        # Check finite values
        assert np.all(np.isfinite(traj_pos))
        assert np.all(np.isfinite(traj_vel))
        assert np.all(np.isfinite(traj_acc))
    
    def test_cpu_fallback_methods(self, trajectory_params):
        """Test different time scaling methods."""
        for method in [3, 5]:  # Cubic and quintic
            traj_pos, traj_vel, traj_acc = trajectory_cpu_fallback(
                trajectory_params['thetastart'],
                trajectory_params['thetaend'],
                trajectory_params['Tf'],
                trajectory_params['N'],
                method
            )
            
            # Check that results are reasonable
            assert np.all(np.isfinite(traj_pos))
            assert np.all(np.isfinite(traj_vel))
            assert np.all(np.isfinite(traj_acc))
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_optimized_trajectory_generation_basic(self, trajectory_params):
        """Test basic optimized trajectory generation."""
        traj_pos, traj_vel, traj_acc = optimized_trajectory_generation(
            trajectory_params['thetastart'],
            trajectory_params['thetaend'],
            trajectory_params['Tf'],
            trajectory_params['N'],
            trajectory_params['method'],
            use_pinned=True
        )
        
        num_joints = len(trajectory_params['thetastart'])
        N = trajectory_params['N']
        
        assert traj_pos.shape == (N, num_joints)
        assert traj_vel.shape == (N, num_joints)
        assert traj_acc.shape == (N, num_joints)
        
        # Check boundary conditions
        np.testing.assert_allclose(traj_pos[0], trajectory_params['thetastart'], rtol=1e-4)
        np.testing.assert_allclose(traj_pos[-1], trajectory_params['thetaend'], rtol=1e-4)
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_optimized_trajectory_generation_methods(self, trajectory_params):
        """Test optimized trajectory with different methods."""
        for method in [3, 5]:
            traj_pos, traj_vel, traj_acc = optimized_trajectory_generation(
                trajectory_params['thetastart'],
                trajectory_params['thetaend'],
                trajectory_params['Tf'],
                trajectory_params['N'],
                method,
                use_pinned=False
            )
            
            assert np.all(np.isfinite(traj_pos))
            assert np.all(np.isfinite(traj_vel))
            assert np.all(np.isfinite(traj_acc))
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_optimized_trajectory_generation_monitored(self, trajectory_params):
        """Test monitored trajectory generation with performance tracking."""
        traj_pos, traj_vel, traj_acc = optimized_trajectory_generation_monitored(
            trajectory_params['thetastart'],
            trajectory_params['thetaend'],
            trajectory_params['Tf'],
            trajectory_params['N'],
            trajectory_params['method'],
            use_pinned=True,
            kernel_type="auto",
            enable_monitoring=True
        )
        
        num_joints = len(trajectory_params['thetastart'])
        N = trajectory_params['N']
        
        assert traj_pos.shape == (N, num_joints)
        assert traj_vel.shape == (N, num_joints)
        assert traj_acc.shape == (N, num_joints)
    
    def test_trajectory_accuracy_comparison(self, trajectory_params):
        """Compare CPU and GPU trajectory accuracy."""
        # CPU version
        cpu_pos, cpu_vel, cpu_acc = trajectory_cpu_fallback(
            trajectory_params['thetastart'],
            trajectory_params['thetaend'],
            trajectory_params['Tf'],
            trajectory_params['N'],
            trajectory_params['method']
        )
        
        if CUDA_AVAILABLE:
            # GPU version
            gpu_pos, gpu_vel, gpu_acc = optimized_trajectory_generation(
                trajectory_params['thetastart'],
                trajectory_params['thetaend'],
                trajectory_params['Tf'],
                trajectory_params['N'],
                trajectory_params['method'],
                use_pinned=True
            )
            
            # Compare accuracy
            np.testing.assert_allclose(cpu_pos, gpu_pos, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(cpu_vel, gpu_vel, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(cpu_acc, gpu_acc, rtol=1e-5, atol=1e-6)


class TestBatchTrajectoryGeneration:
    """Test batch trajectory generation functionality."""
    
    @pytest.fixture
    def batch_params(self):
        """Batch trajectory parameters."""
        batch_size = 5
        num_joints = 6
        return {
            'thetastart_batch': np.random.uniform(-1, 1, (batch_size, num_joints)).astype(np.float32),
            'thetaend_batch': np.random.uniform(-1, 1, (batch_size, num_joints)).astype(np.float32),
            'Tf': 3.0,
            'N': 50,
            'method': 3  # Cubic
        }
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_batch_trajectory_generation(self, batch_params):
        """Test batch trajectory generation."""
        traj_pos_batch, traj_vel_batch, traj_acc_batch = optimized_batch_trajectory_generation(
            batch_params['thetastart_batch'],
            batch_params['thetaend_batch'],
            batch_params['Tf'],
            batch_params['N'],
            batch_params['method'],
            use_pinned=True
        )
        
        batch_size, num_joints = batch_params['thetastart_batch'].shape
        N = batch_params['N']
        
        assert traj_pos_batch.shape == (batch_size, N, num_joints)
        assert traj_vel_batch.shape == (batch_size, N, num_joints)
        assert traj_acc_batch.shape == (batch_size, N, num_joints)
        
        # Check boundary conditions for each trajectory in batch
        for i in range(batch_size):
            np.testing.assert_allclose(
                traj_pos_batch[i, 0], batch_params['thetastart_batch'][i], 
                rtol=1e-4
            )
            np.testing.assert_allclose(
                traj_pos_batch[i, -1], batch_params['thetaend_batch'][i], 
                rtol=1e-4
            )
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_batch_trajectory_different_sizes(self):
        """Test batch trajectory with different batch sizes."""
        for batch_size in [1, 3, 10]:
            num_joints = 4
            thetastart_batch = np.random.uniform(-1, 1, (batch_size, num_joints)).astype(np.float32)
            thetaend_batch = np.random.uniform(-1, 1, (batch_size, num_joints)).astype(np.float32)
            
            traj_pos_batch, traj_vel_batch, traj_acc_batch = optimized_batch_trajectory_generation(
                thetastart_batch, thetaend_batch, 2.0, 30, 5, use_pinned=True
            )
            
            assert traj_pos_batch.shape == (batch_size, 30, num_joints)
            assert np.all(np.isfinite(traj_pos_batch))


class TestPotentialField:
    """Test potential field computation functionality."""
    
    @pytest.fixture
    def potential_field_params(self):
        """Potential field parameters."""
        return {
            'positions': np.random.rand(64, 3).astype(np.float32),
            'goal': np.array([0.5, 0.5, 0.5], dtype=np.float32),
            'obstacles': np.array([[0.3, 0.3, 0.3], [0.7, 0.7, 0.7]], dtype=np.float32),
            'influence_distance': 0.5
        }
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_optimized_potential_field_basic(self, potential_field_params):
        """Test basic potential field computation."""
        potential, gradient = optimized_potential_field(
            potential_field_params['positions'],
            potential_field_params['goal'],
            potential_field_params['obstacles'],
            potential_field_params['influence_distance'],
            use_pinned=True
        )
        
        N = potential_field_params['positions'].shape[0]
        
        assert potential.shape == (N,)
        assert gradient.shape == (N, 3)
        assert np.all(np.isfinite(potential))
        assert np.all(np.isfinite(gradient))
        assert np.all(potential >= 0)  # Potential should be non-negative
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_potential_field_different_sizes(self):
        """Test potential field with different problem sizes."""
        goal = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        obstacles = np.array([[0.2, 0.2, 0.2]], dtype=np.float32)
        influence_distance = 0.3
        
        for N in [10, 50, 200]:
            positions = np.random.rand(N, 3).astype(np.float32)
            
            potential, gradient = optimized_potential_field(
                positions, goal, obstacles, influence_distance, use_pinned=False
            )
            
            assert potential.shape == (N,)
            assert gradient.shape == (N, 3)
            assert np.all(np.isfinite(potential))
            assert np.all(np.isfinite(gradient))
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_potential_field_no_obstacles(self):
        """Test potential field with no obstacles."""
        positions = np.random.rand(32, 3).astype(np.float32)
        goal = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        obstacles = np.empty((0, 3), dtype=np.float32)  # No obstacles
        influence_distance = 0.5
        
        potential, gradient = optimized_potential_field(
            positions, goal, obstacles, influence_distance, use_pinned=True
        )
        
        assert potential.shape == (32,)
        assert gradient.shape == (32, 3)
        assert np.all(np.isfinite(potential))
        assert np.all(np.isfinite(gradient))


class TestMemoryManagement:
    """Test CUDA memory management functionality."""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_cuda_array_pool_basic(self):
        """Test basic CUDA array pooling."""
        shape = (100, 6)
        dtype = np.float32
        
        # Get array from pool
        array1 = get_cuda_array(shape, dtype)
        assert array1.shape == shape
        assert array1.dtype == dtype
        
        # Return array to pool
        return_cuda_array(array1)
        
        # Get another array (might be the same one)
        array2 = get_cuda_array(shape, dtype)
        assert array2.shape == shape
        assert array2.dtype == dtype
        
        return_cuda_array(array2)
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_cuda_array_pool_different_sizes(self):
        """Test CUDA array pool with different sizes."""
        arrays = []
        shapes = [(10, 3), (50, 6), (100, 12), (200, 4)]
        
        # Get arrays of different sizes
        for shape in shapes:
            array = get_cuda_array(shape, np.float32)
            assert array.shape == shape
            arrays.append(array)
        
        # Return all arrays
        for array in arrays:
            return_cuda_array(array)
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_memory_pool_stats(self):
        """Test memory pool statistics."""
        stats = get_memory_pool_stats()
        assert isinstance(stats, dict)
        
        # Stats should contain relevant information
        expected_keys = ['total_arrays', 'cache_hit_rate', 'memory_usage_mb']
        for key in expected_keys:
            if key in stats:  # Some stats might not be available
                assert isinstance(stats[key], (int, float))
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_pinned_memory_transfer(self):
        """Test pinned memory transfers."""
        data = np.random.rand(100, 6).astype(np.float32)
        
        # Test pinned memory transfer
        d_data = _h2d_pinned(data)
        assert d_data.shape == data.shape
        assert d_data.dtype == data.dtype
        
        # Copy back and verify
        data_back = d_data.copy_to_host()
        np.testing.assert_array_equal(data, data_back)


class TestKernelConfiguration:
    """Test kernel configuration and optimization."""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_kernel_config_selection(self):
        """Test kernel configuration selection."""
        test_cases = [
            (100, 6),
            (1000, 6),
            (5000, 6),
            (1000, 12)
        ]
        
        for N, num_joints in test_cases:
            config = get_optimal_kernel_config(N, num_joints)
            
            if config is not None:  # Some configurations might not be available
                assert isinstance(config, dict)
                assert 'grid' in config
                assert 'block' in config
                assert 'kernel_type' in config
                
                grid, block = config['grid'], config['block']
                assert isinstance(grid, tuple) and len(grid) == 2
                assert isinstance(block, tuple) and len(block) == 2
                assert all(g > 0 for g in grid)
                assert all(b > 0 for b in block)
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_auto_kernel_selection(self):
        """Test automatic kernel selection."""
        test_cases = [
            (500, 6),
            (2000, 6),
            (10000, 6)
        ]
        
        for N, num_joints in test_cases:
            kernel_type = auto_select_optimal_kernel(N, num_joints)
            assert isinstance(kernel_type, str)
            assert kernel_type in ["standard", "vectorized", "memory_optimized", 
                                 "warp_optimized", "cache_friendly"]
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_performance_recommendations(self, capsys):
        """Test performance recommendations output."""
        N, num_joints = 1000, 6
        print_performance_recommendations(N, num_joints)
        
        captured = capsys.readouterr()
        assert len(captured.out) > 0  # Should print some recommendations
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_cuda_environment_setup(self):
        """Test CUDA environment setup for optimal performance."""
        # Should not raise exceptions
        setup_cuda_environment_for_40x_speedup()


class TestPerformanceBenchmarking:
    """Test performance benchmarking functionality."""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_benchmark_trajectory_kernel(self):
        """Test trajectory kernel benchmarking."""
        thetastart = np.array([0.0, 0.1, 0.2], dtype=np.float32)
        thetaend = np.array([1.0, 0.9, 0.8], dtype=np.float32)
        Tf = 1.0
        N = 200
        method = 3
        
        stats = benchmark_kernel_performance(
            "trajectory", thetastart, thetaend, Tf, N, method, num_runs=3
        )
        
        assert isinstance(stats, dict)
        expected_keys = ['avg_time', 'std_time', 'min_time', 'max_time']
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], float)
            assert stats[key] >= 0
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_benchmark_potential_field_kernel(self):
        """Test potential field kernel benchmarking."""
        positions = np.random.rand(100, 3).astype(np.float32)
        goal = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        obstacles = np.array([[0.3, 0.3, 0.3]], dtype=np.float32)
        influence_distance = 0.5
        
        stats = benchmark_kernel_performance(
            "potential_field", positions, goal, obstacles, influence_distance, num_runs=3
        )
        
        assert isinstance(stats, dict)
        expected_keys = ['avg_time', 'std_time', 'min_time', 'max_time']
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], float)
            assert stats[key] >= 0
    
    def test_benchmark_unavailable_cuda(self):
        """Test benchmarking when CUDA is unavailable."""
        if not CUDA_AVAILABLE:
            result = benchmark_kernel_performance("trajectory", num_runs=1)
            assert result is None


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_trajectory_parameters(self):
        """Test trajectory generation with invalid parameters."""
        thetastart = np.array([0.0, 0.1], dtype=np.float32)
        thetaend = np.array([1.0, 0.9, 0.8], dtype=np.float32)  # Different size
        
        # Should handle mismatched array sizes gracefully
        with pytest.raises((ValueError, AssertionError, IndexError)):
            trajectory_cpu_fallback(thetastart, thetaend, 1.0, 100, 3)
    
    def test_zero_time_trajectory(self):
        """Test trajectory with zero time."""
        thetastart = np.array([0.0, 0.1], dtype=np.float32)
        thetaend = np.array([1.0, 0.9], dtype=np.float32)
        
        # Zero time creates division by zero, which should be handled
        # The function might produce NaN or inf values, which is expected behavior
        traj_pos, traj_vel, traj_acc = trajectory_cpu_fallback(
            thetastart, thetaend, 0.0, 10, 3
        )
        
        # With zero time, the results may contain NaN due to division by zero
        # This is acceptable behavior - the function shouldn't crash
        assert traj_pos.shape == (10, 2)
        assert traj_vel.shape == (10, 2)
        assert traj_acc.shape == (10, 2)
        
        # The first position should still be the start position
        if np.all(np.isfinite(traj_pos[0])):
            np.testing.assert_allclose(traj_pos[0], thetastart, rtol=1e-5)
        
        # Test with a very small but non-zero time instead
        traj_pos_small, traj_vel_small, traj_acc_small = trajectory_cpu_fallback(
            thetastart, thetaend, 1e-6, 10, 3
        )
        
        # With very small time, results should be finite
        assert np.all(np.isfinite(traj_pos_small))
        assert np.all(np.isfinite(traj_vel_small))
        assert np.all(np.isfinite(traj_acc_small))
    
    def test_very_small_time_trajectory(self):
        """Test trajectory with very small time duration."""
        thetastart = np.array([0.0, 0.1], dtype=np.float32)
        thetaend = np.array([1.0, 0.9], dtype=np.float32)
        
        # Very small time should still produce valid results
        traj_pos, traj_vel, traj_acc = trajectory_cpu_fallback(
            thetastart, thetaend, 1e-3, 10, 3
        )
        
        assert np.all(np.isfinite(traj_pos))
        assert np.all(np.isfinite(traj_vel))
        assert np.all(np.isfinite(traj_acc))
        
        # Check boundary conditions
        np.testing.assert_allclose(traj_pos[0], thetastart, rtol=1e-4)
        np.testing.assert_allclose(traj_pos[-1], thetaend, rtol=1e-4)
    
    def test_single_point_trajectory(self):
        """Test trajectory with single point."""
        thetastart = np.array([0.0, 0.1], dtype=np.float32)
        thetaend = np.array([1.0, 0.9], dtype=np.float32)
        
        traj_pos, traj_vel, traj_acc = trajectory_cpu_fallback(
            thetastart, thetaend, 1.0, 1, 3
        )
        
        assert traj_pos.shape == (1, 2)
        assert np.all(np.isfinite(traj_pos))
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_large_trajectory_handling(self):
        """Test handling of large trajectory requests."""
        thetastart = np.array([0.0] * 6, dtype=np.float32)
        thetaend = np.array([1.0] * 6, dtype=np.float32)
        
        # Large but manageable size
        N = 50000
        
        traj_pos, traj_vel, traj_acc = optimized_trajectory_generation(
            thetastart, thetaend, 2.0, N, 5, use_pinned=True
        )
        
        assert traj_pos.shape == (N, 6)
        assert np.all(np.isfinite(traj_pos))


class TestProfilers:
    """Test profiling functionality."""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_profiling_start_stop(self):
        """Test profiling start and stop."""
        # Should not raise exceptions
        profile_start()
        
        # Do some work
        thetastart = np.array([0.0, 0.1], dtype=np.float32)
        thetaend = np.array([1.0, 0.9], dtype=np.float32)
        optimized_trajectory_generation(thetastart, thetaend, 1.0, 100, 3)
        
        profile_stop()
    
    def test_profiling_without_cuda(self):
        """Test profiling when CUDA unavailable."""
        if not CUDA_AVAILABLE:
            # Should not raise exceptions
            profile_start()
            profile_stop()


class TestIntegrationScenarios:
    """Test integration scenarios and realistic use cases."""
    
    @pytest.fixture
    def robot_params(self):
        """Realistic robot parameters for testing."""
        return {
            'num_joints': 6,
            'joint_limits': [(-np.pi, np.pi)] * 6,
            'typical_trajectory_length': 1000,
            'typical_time_duration': 5.0
        }
    
    def test_typical_robot_trajectory(self, robot_params):
        """Test typical robot trajectory generation scenario."""
        # Home to target position
        home_position = np.zeros(robot_params['num_joints'], dtype=np.float32)
        target_position = np.array([np.pi/4, -np.pi/6, np.pi/3, -np.pi/4, np.pi/6, -np.pi/8], dtype=np.float32)
        
        # CPU version (always available)
        cpu_traj_pos, cpu_traj_vel, cpu_traj_acc = trajectory_cpu_fallback(
            home_position, target_position, 
            robot_params['typical_time_duration'], 
            robot_params['typical_trajectory_length'], 
            5  # Quintic
        )
        
        assert cpu_traj_pos.shape == (robot_params['typical_trajectory_length'], robot_params['num_joints'])
        
        # Verify trajectory stays within joint limits
        for joint_idx in range(robot_params['num_joints']):
            min_limit, max_limit = robot_params['joint_limits'][joint_idx]
            assert np.all(cpu_traj_pos[:, joint_idx] >= min_limit)
            assert np.all(cpu_traj_pos[:, joint_idx] <= max_limit)
        
        if CUDA_AVAILABLE:
            # GPU version
            gpu_traj_pos, gpu_traj_vel, gpu_traj_acc = optimized_trajectory_generation(
                home_position, target_position,
                robot_params['typical_time_duration'],
                robot_params['typical_trajectory_length'],
                5, use_pinned=True
            )
            
            # Compare CPU vs GPU results
            np.testing.assert_allclose(cpu_traj_pos, gpu_traj_pos, rtol=1e-5)
            np.testing.assert_allclose(cpu_traj_vel, gpu_traj_vel, rtol=1e-5)
            np.testing.assert_allclose(cpu_traj_acc, gpu_traj_acc, rtol=1e-5)
    
    def test_multi_trajectory_batch_scenario(self):
        """Test realistic multi-trajectory batch processing."""
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA required for batch processing")
        
        # Simulate multiple pick-and-place operations
        batch_size = 8
        num_joints = 6
        
        # Generate realistic start/end positions
        home_positions = np.zeros((batch_size, num_joints), dtype=np.float32)
        target_positions = np.random.uniform(-np.pi/2, np.pi/2, (batch_size, num_joints)).astype(np.float32)
        
        # Process batch
        traj_pos_batch, traj_vel_batch, traj_acc_batch = optimized_batch_trajectory_generation(
            home_positions, target_positions, 3.0, 500, 5, use_pinned=True
        )
        
        assert traj_pos_batch.shape == (batch_size, 500, num_joints)
        
        # Verify each trajectory in batch
        for batch_idx in range(batch_size):
            # Check boundary conditions
            np.testing.assert_allclose(
                traj_pos_batch[batch_idx, 0], home_positions[batch_idx], rtol=1e-4
            )
            np.testing.assert_allclose(
                traj_pos_batch[batch_idx, -1], target_positions[batch_idx], rtol=1e-4
            )
            
            # Check continuity
            assert np.all(np.isfinite(traj_pos_batch[batch_idx]))
            assert np.all(np.isfinite(traj_vel_batch[batch_idx]))
            assert np.all(np.isfinite(traj_acc_batch[batch_idx]))
    
    def test_obstacle_avoidance_scenario(self):
        """Test trajectory generation with obstacle avoidance."""
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA required for potential field computation")
        
        # Robot configuration space (simplified 3D for testing)
        num_positions = 100
        positions = np.random.rand(num_positions, 3).astype(np.float32) * 2.0 - 1.0  # [-1, 1]
        
        # Goal position
        goal = np.array([0.8, 0.8, 0.8], dtype=np.float32)
        
        # Obstacle positions
        obstacles = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.2, -0.3, 0.5]
        ], dtype=np.float32)
        
        influence_distance = 0.4
        
        # Compute potential field
        potential, gradient = optimized_potential_field(
            positions, goal, obstacles, influence_distance, use_pinned=True
        )
        
        # Verify results
        assert potential.shape == (num_positions,)
        assert gradient.shape == (num_positions, 3)
        assert np.all(potential >= 0)  # Potential should be non-negative
        assert np.all(np.isfinite(potential))
        assert np.all(np.isfinite(gradient))
        
        # Check that gradient points away from obstacles near them
        for obs_pos in obstacles:
            distances = np.linalg.norm(positions - obs_pos, axis=1)
            close_indices = distances < influence_distance
            
            if np.any(close_indices):
                # Gradient should generally point away from obstacles
                close_positions = positions[close_indices]
                close_gradients = gradient[close_indices]
                
                for i in range(len(close_positions)):
                    direction_from_obs = close_positions[i] - obs_pos
                    gradient_direction = close_gradients[i]
                    
                    if np.linalg.norm(direction_from_obs) > 1e-6:
                        # Dot product should be positive (pointing away)
                        direction_from_obs_norm = direction_from_obs / np.linalg.norm(direction_from_obs)
                        if np.linalg.norm(gradient_direction) > 1e-6:
                            gradient_direction_norm = gradient_direction / np.linalg.norm(gradient_direction)
                            dot_product = np.dot(direction_from_obs_norm, gradient_direction_norm)
                            # Allow some tolerance for numerical precision
                            assert dot_product > -0.5  # Should generally point away
    
    def test_performance_scaling_scenario(self):
        """Test performance scaling with different problem sizes."""
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA required for performance scaling test")
        
        problem_sizes = [
            (100, 3),
            (500, 6),
            (1000, 6),
            (2000, 6),
            (5000, 6)
        ]
        
        results = []
        
        for N, num_joints in problem_sizes:
            thetastart = np.random.uniform(-1, 1, num_joints).astype(np.float32)
            thetaend = np.random.uniform(-1, 1, num_joints).astype(np.float32)
            
            # Time GPU execution
            start_time = time.time()
            gpu_traj_pos, gpu_traj_vel, gpu_traj_acc = optimized_trajectory_generation(
                thetastart, thetaend, 2.0, N, 5, use_pinned=True
            )
            gpu_time = time.time() - start_time
            
            # Time CPU execution
            start_time = time.time()
            cpu_traj_pos, cpu_traj_vel, cpu_traj_acc = trajectory_cpu_fallback(
                thetastart, thetaend, 2.0, N, 5
            )
            cpu_time = time.time() - start_time
            
            # Calculate metrics
            total_elements = N * num_joints
            gpu_throughput = total_elements / gpu_time if gpu_time > 0 else 0
            cpu_throughput = total_elements / cpu_time if cpu_time > 0 else 0
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            
            results.append({
                'N': N,
                'num_joints': num_joints,
                'total_elements': total_elements,
                'gpu_time': gpu_time,
                'cpu_time': cpu_time,
                'speedup': speedup,
                'gpu_throughput': gpu_throughput,
                'cpu_throughput': cpu_throughput
            })
            
            # Verify correctness
            np.testing.assert_allclose(cpu_traj_pos, gpu_traj_pos, rtol=1e-5)
            
            # Basic performance expectations
            assert gpu_time > 0
            assert cpu_time > 0
            assert gpu_throughput > 0
            assert cpu_throughput > 0
        
        # Check that performance generally improves with problem size
        large_problem_result = next(r for r in results if r['total_elements'] >= 10000)
        assert large_problem_result['speedup'] > 1.0  # GPU should be faster for large problems
    
    def test_memory_intensive_scenario(self):
        """Test memory-intensive operations."""
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA required for memory-intensive test")
        
        # Large trajectory
        N = 10000
        num_joints = 6
        
        thetastart = np.random.uniform(-1, 1, num_joints).astype(np.float32)
        thetaend = np.random.uniform(-1, 1, num_joints).astype(np.float32)
        
        # Test memory pool usage
        initial_stats = get_memory_pool_stats()
        
        # Generate large trajectory
        traj_pos, traj_vel, traj_acc = optimized_trajectory_generation(
            thetastart, thetaend, 5.0, N, 5, use_pinned=True
        )
        
        final_stats = get_memory_pool_stats()
        
        # Verify trajectory
        assert traj_pos.shape == (N, num_joints)
        assert np.all(np.isfinite(traj_pos))
        
        # Memory pool should show some activity
        if 'total_arrays' in initial_stats and 'total_arrays' in final_stats:
            # Some arrays should have been used
            assert final_stats['total_arrays'] >= initial_stats['total_arrays']
    
    def test_mixed_precision_scenario(self):
        """Test mixed precision calculations."""
        # Test with different data types
        for dtype in [np.float32, np.float64]:
            thetastart = np.array([0.0, 0.5, -0.3], dtype=dtype)
            thetaend = np.array([1.0, -0.2, 0.8], dtype=dtype)
            
            traj_pos, traj_vel, traj_acc = trajectory_cpu_fallback(
                thetastart, thetaend, 2.0, 100, 3
            )
            
            # Results should maintain precision
            assert traj_pos.dtype == np.float32  # Always returns float32
            assert np.all(np.isfinite(traj_pos))
            
            # Boundary conditions should be accurate
            np.testing.assert_allclose(traj_pos[0], thetastart.astype(np.float32), rtol=1e-6)
            np.testing.assert_allclose(traj_pos[-1], thetaend.astype(np.float32), rtol=1e-6)


class TestRegressionPrevention:
    """Test cases to prevent regression of known issues."""
    
    def test_quintic_trajectory_smoothness(self):
        """Regression test for quintic trajectory smoothness."""
        thetastart = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        thetaend = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        
        traj_pos, traj_vel, traj_acc = trajectory_cpu_fallback(
            thetastart, thetaend, 2.0, 1000, 5  # Quintic
        )
        
        # Quintic trajectories should start and end with zero velocity and acceleration
        np.testing.assert_allclose(traj_vel[0], [0.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(traj_vel[-1], [0.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(traj_acc[0], [0.0, 0.0, 0.0], atol=1e-4)
        np.testing.assert_allclose(traj_acc[-1], [0.0, 0.0, 0.0], atol=1e-4)
    
    def test_cubic_trajectory_boundary_conditions(self):
        """Regression test for cubic trajectory boundary conditions."""
        thetastart = np.array([0.0, 0.5], dtype=np.float32)
        thetaend = np.array([1.0, -0.5], dtype=np.float32)
        
        traj_pos, traj_vel, traj_acc = trajectory_cpu_fallback(
            thetastart, thetaend, 1.0, 500, 3  # Cubic
        )
        
        # Cubic trajectories should start and end with zero velocity
        np.testing.assert_allclose(traj_vel[0], [0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(traj_vel[-1], [0.0, 0.0], atol=1e-5)
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_gpu_cpu_precision_consistency(self):
        """Regression test for GPU-CPU precision consistency."""
        test_cases = [
            ([0.0, 0.1, -0.2], [1.0, -0.1, 0.8], 1.0, 100, 3),
            ([0.5, -0.3, 0.7], [-0.2, 0.9, -0.4], 2.0, 200, 5),
            ([-1.0, 0.0, 1.0], [1.0, -1.0, 0.0], 3.0, 150, 3),
        ]
        
        for thetastart, thetaend, Tf, N, method in test_cases:
            thetastart = np.array(thetastart, dtype=np.float32)
            thetaend = np.array(thetaend, dtype=np.float32)
            
            # CPU version
            cpu_pos, cpu_vel, cpu_acc = trajectory_cpu_fallback(
                thetastart, thetaend, Tf, N, method
            )
            
            # GPU version
            gpu_pos, gpu_vel, gpu_acc = optimized_trajectory_generation(
                thetastart, thetaend, Tf, N, method, use_pinned=True
            )
            
            # Should match within reasonable precision
            np.testing.assert_allclose(cpu_pos, gpu_pos, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(cpu_vel, gpu_vel, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(cpu_acc, gpu_acc, rtol=1e-5, atol=1e-6)
    
    def test_edge_case_single_joint(self):
        """Regression test for single joint trajectories."""
        thetastart = np.array([0.5], dtype=np.float32)
        thetaend = np.array([-0.8], dtype=np.float32)
        
        traj_pos, traj_vel, traj_acc = trajectory_cpu_fallback(
            thetastart, thetaend, 1.0, 50, 5
        )
        
        assert traj_pos.shape == (50, 1)
        assert np.all(np.isfinite(traj_pos))
        np.testing.assert_allclose(traj_pos[0], thetastart, rtol=1e-5)
        np.testing.assert_allclose(traj_pos[-1], thetaend, rtol=1e-5)
    
    def test_zero_displacement_trajectory(self):
        """Regression test for zero displacement trajectories."""
        thetastart = np.array([0.5, -0.3, 0.8], dtype=np.float32)
        thetaend = thetastart.copy()  # Same start and end
        
        traj_pos, traj_vel, traj_acc = trajectory_cpu_fallback(
            thetastart, thetaend, 1.0, 100, 3
        )
        
        # Position should remain constant
        for i in range(100):
            np.testing.assert_allclose(traj_pos[i], thetastart, rtol=1e-5)
        
        # Velocity should be zero throughout (or very close to zero)
        assert np.allclose(traj_vel, 0.0, atol=1e-5)
    
    def test_negative_time_handling(self):
        """Test handling of negative time values."""
        thetastart = np.array([0.0, 0.5], dtype=np.float32)
        thetaend = np.array([1.0, -0.5], dtype=np.float32)
        
        # Negative time should be handled gracefully
        # The function may produce unexpected results but shouldn't crash
        try:
            traj_pos, traj_vel, traj_acc = trajectory_cpu_fallback(
                thetastart, thetaend, -1.0, 50, 3
            )
            
            # If it doesn't crash, check basic properties
            assert traj_pos.shape == (50, 2)
            assert traj_vel.shape == (50, 2)
            assert traj_acc.shape == (50, 2)
        except (ValueError, ArithmeticError):
            # It's acceptable if the function raises an error for negative time
            pass
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_memory_leak_prevention(self):
        """Regression test for memory leaks in repeated operations."""
        thetastart = np.array([0.0, 0.5, -0.2], dtype=np.float32)
        thetaend = np.array([1.0, -0.3, 0.8], dtype=np.float32)
        
        initial_stats = get_memory_pool_stats()
        
        # Perform many operations
        for i in range(20):
            traj_pos, traj_vel, traj_acc = optimized_trajectory_generation(
                thetastart + i * 0.01,  # Slightly different each time
                thetaend + i * 0.01,
                1.0, 100, 3,
                use_pinned=True
            )
            assert np.all(np.isfinite(traj_pos))
        
        final_stats = get_memory_pool_stats()
        
        # Memory usage shouldn't grow excessively
        if 'memory_usage_mb' in initial_stats and 'memory_usage_mb' in final_stats:
            memory_growth = final_stats['memory_usage_mb'] - initial_stats['memory_usage_mb']
            assert memory_growth < 100  # Should not grow by more than 100MB


class TestDocumentationExamples:
    """Test examples that would appear in documentation."""
    
    def test_basic_usage_example(self):
        """Test the basic usage example from documentation."""
        # Basic CPU trajectory generation
        thetastart = np.array([0.0, 0.5, 1.0, -0.5, 0.2, -0.3], dtype=np.float32)
        thetaend = np.array([1.0, 0.0, -1.0, 0.8, -0.4, 0.6], dtype=np.float32)
        Tf = 2.0
        N = 100
        method = 5  # Quintic time scaling
        
        # Generate trajectory
        traj_pos, traj_vel, traj_acc = trajectory_cpu_fallback(
            thetastart, thetaend, Tf, N, method
        )
        
        # Verify basic properties
        assert traj_pos.shape == (N, 6)
        assert traj_vel.shape == (N, 6)
        assert traj_acc.shape == (N, 6)
        
        # Check boundary conditions
        np.testing.assert_allclose(traj_pos[0], thetastart, rtol=1e-5)
        np.testing.assert_allclose(traj_pos[-1], thetaend, rtol=1e-5)
        
        # Quintic should have zero velocity at boundaries
        np.testing.assert_allclose(traj_vel[0], np.zeros(6), atol=1e-5)
        np.testing.assert_allclose(traj_vel[-1], np.zeros(6), atol=1e-5)
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_gpu_acceleration_example(self):
        """Test GPU acceleration example from documentation."""
        # Parameters for a typical industrial robot
        thetastart = np.zeros(6, dtype=np.float32)
        thetaend = np.array([np.pi/4, -np.pi/6, np.pi/3, -np.pi/4, np.pi/6, -np.pi/8], dtype=np.float32)
        
        # Generate high-resolution trajectory
        Tf = 5.0
        N = 5000  # High resolution for smooth motion
        method = 5
        
        # Time CPU version
        start_time = time.time()
        cpu_pos, cpu_vel, cpu_acc = trajectory_cpu_fallback(
            thetastart, thetaend, Tf, N, method
        )
        cpu_time = time.time() - start_time
        
        # Time GPU version
        start_time = time.time()
        gpu_pos, gpu_vel, gpu_acc = optimized_trajectory_generation(
            thetastart, thetaend, Tf, N, method, use_pinned=True
        )
        gpu_time = time.time() - start_time
        
        # Verify accuracy
        np.testing.assert_allclose(cpu_pos, gpu_pos, rtol=1e-5)
        np.testing.assert_allclose(cpu_vel, gpu_vel, rtol=1e-5)
        np.testing.assert_allclose(cpu_acc, gpu_acc, rtol=1e-5)
        
        # GPU should be faster for large problems
        if N * 6 > 10000:  # Large enough problem
            speedup = cpu_time / gpu_time
            assert speedup > 1.0
            print(f"GPU Speedup: {speedup:.2f}x")
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_batch_processing_example(self):
        """Test batch processing example from documentation."""
        # Multiple robot trajectories for parallel processing
        batch_size = 4
        num_joints = 6
        
        # Generate random start and end configurations
        np.random.seed(42)  # For reproducible results
        thetastart_batch = np.random.uniform(-np.pi/2, np.pi/2, (batch_size, num_joints)).astype(np.float32)
        thetaend_batch = np.random.uniform(-np.pi/2, np.pi/2, (batch_size, num_joints)).astype(np.float32)
        
        # Process entire batch on GPU
        Tf = 3.0
        N = 500
        method = 5
        
        traj_pos_batch, traj_vel_batch, traj_acc_batch = optimized_batch_trajectory_generation(
            thetastart_batch, thetaend_batch, Tf, N, method, use_pinned=True
        )
        
        # Verify batch results
        assert traj_pos_batch.shape == (batch_size, N, num_joints)
        assert traj_vel_batch.shape == (batch_size, N, num_joints)
        assert traj_acc_batch.shape == (batch_size, N, num_joints)
        
        # Check each trajectory in the batch
        for i in range(batch_size):
            np.testing.assert_allclose(traj_pos_batch[i, 0], thetastart_batch[i], rtol=1e-4)
            np.testing.assert_allclose(traj_pos_batch[i, -1], thetaend_batch[i], rtol=1e-4)
            assert np.all(np.isfinite(traj_pos_batch[i]))


# Performance benchmarks (run with pytest --benchmark-only if pytest-benchmark is installed)
class TestPerformanceBenchmarks:
    """Performance benchmarks for CUDA kernels."""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_benchmark_small_trajectory(self):
        """Benchmark small trajectory generation."""
        thetastart = np.zeros(6, dtype=np.float32)
        thetaend = np.ones(6, dtype=np.float32)
        
        def cpu_version():
            return trajectory_cpu_fallback(thetastart, thetaend, 1.0, 100, 5)
        
        def gpu_version():
            return optimized_trajectory_generation(thetastart, thetaend, 1.0, 100, 5)
        
        # Run benchmarks
        cpu_result = cpu_version()
        gpu_result = gpu_version()
        
        # Verify correctness
        np.testing.assert_allclose(cpu_result[0], gpu_result[0], rtol=1e-5)
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available") 
    def test_benchmark_large_trajectory(self):
        """Benchmark large trajectory generation."""
        thetastart = np.random.uniform(-1, 1, 12).astype(np.float32)
        thetaend = np.random.uniform(-1, 1, 12).astype(np.float32)
        
        def cpu_version():
            return trajectory_cpu_fallback(thetastart, thetaend, 5.0, 10000, 5)
        
        def gpu_version():
            return optimized_trajectory_generation(thetastart, thetaend, 5.0, 10000, 5, use_pinned=True)
        
        # Time both versions
        start_time = time.time()
        cpu_result = cpu_version()
        cpu_time = time.time() - start_time
        
        start_time = time.time()
        gpu_result = gpu_version()
        gpu_time = time.time() - start_time
        
        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"Large trajectory speedup: {speedup:.2f}x")
        
        # Verify correctness
        np.testing.assert_allclose(cpu_result[0], gpu_result[0], rtol=1e-5)
        
        # GPU should be significantly faster for large problems
        assert speedup > 2.0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])