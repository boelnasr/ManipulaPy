
# ManipulaPy Path Planning Performance Benchmark Report

## System Information
- CPU Cores: 16
- Memory: 31.1 GB
- CUDA Available: True
- GPU Multiprocessors: 30
- Max Threads per Block: 1024
- Max Shared Memory: 49152 bytes

## Benchmark Configuration
- Test Name: ManipulaPy_Performance_Benchmark
- Trajectory Points: [10000, 50000, 100000]
- Joint Counts: [3, 6]
- Batch Sizes: [1, 2, 3]
- Runs per Test: 3
- Timeout: 60.0s

## Results Summary
- Total Tests: 36
- Successful Tests: 33
- Failed Tests: 3

## Performance Analysis

### Joint Trajectory
- Mean Speedup: 2.29x
- Max Speedup: 7.96x
- Min Speedup: 0.10x
- Standard Deviation: 2.65
- Sample Size: 6

### Inverse Dynamics
- Mean Speedup: 3624.62x
- Max Speedup: 5562.57x
- Min Speedup: 393.35x
- Standard Deviation: 2299.96
- Sample Size: 3

### Cartesian Trajectory
- Mean Speedup: 1.02x
- Max Speedup: 1.05x
- Min Speedup: 0.97x
- Standard Deviation: 0.04
- Sample Size: 3

## Overall Statistics
- CPU Mean Time: 6.8762s
- GPU Mean Time: 0.5283s
- Performance Factor: 13.02×
- CPU Success Rate: 100.0%
- GPU Success Rate: 100.0%
- Total CPU Tests: 12
- Total GPU Tests: 21

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
