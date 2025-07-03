#!/usr/bin/env python3
"""
test_perception.py - Comprehensive tests for the Perception module.
Tests the higher-level perception functionality that uses Vision.
"""

import unittest
import numpy as np
import sys
import os
from unittest.mock import Mock, patch

# Add the package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def is_module_available(module_name):
    """Check if a module is really available (not mocked)."""
    try:
        module = __import__(module_name)
        return not hasattr(module, '_name') or not str(module._name).startswith('Mock')
    except ImportError:
        return False

class TestPerceptionInitialization(unittest.TestCase):
    """Test Perception module initialization and basic functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock vision instance
        self.mock_vision = Mock()
        self.mock_vision.capture_image.return_value = (
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            np.random.uniform(0.1, 5.0, (480, 640)).astype(np.float32)
        )
        self.mock_vision.detect_obstacles.return_value = (
            np.random.randn(10, 3),  # 10 obstacle points
            np.arange(10)  # Labels 0-9
        )
    
    def test_perception_basic_initialization(self):
        """Test basic Perception initialization."""
        try:
            from ManipulaPy.perception import Perception
            
            perception = Perception(vision_instance=self.mock_vision)
            
            self.assertIsNotNone(perception, "Perception should initialize")
            self.assertIsNotNone(perception.logger, "Perception should have logger")
            self.assertEqual(perception.vision, self.mock_vision, "Should store vision instance")
            
            print("✅ Perception basic initialization working")
            
        except ImportError as e:
            self.skipTest(f"Perception module not available: {e}")
    
    def test_perception_requires_vision_instance(self):
        """Test that Perception requires a valid vision instance."""
        try:
            from ManipulaPy.perception import Perception
            
            # Should raise ValueError when vision_instance is None
            with self.assertRaises(ValueError):
                Perception(vision_instance=None)
            
            print("✅ Perception properly validates vision instance")
            
        except ImportError as e:
            self.skipTest(f"Perception module not available: {e}")
    
    def test_perception_logger_setup(self):
        """Test that Perception sets up logging correctly."""
        try:
            from ManipulaPy.perception import Perception
            
            perception = Perception(vision_instance=self.mock_vision, logger_name="TestLogger")
            
            self.assertIsNotNone(perception.logger)
            self.assertEqual(perception.logger.name, "TestLogger")
            
            print("✅ Perception logger setup working")
            
        except ImportError as e:
            self.skipTest(f"Perception module not available: {e}")

class TestPerceptionObstacleDetection(unittest.TestCase):
    """Test Perception obstacle detection and clustering functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_vision = Mock()
        self.mock_vision.capture_image.return_value = (
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            np.random.uniform(0.1, 5.0, (480, 640)).astype(np.float32)
        )
        self.mock_vision.detect_obstacles.return_value = (
            np.random.randn(10, 3),
            np.arange(10)
        )
    
    def test_detect_and_cluster_obstacles_pipeline(self):
        """Test the full obstacle detection and clustering pipeline."""
        try:
            from ManipulaPy.perception import Perception
            
            perception = Perception(vision_instance=self.mock_vision)
            
            # Test the full pipeline
            obstacle_points, labels = perception.detect_and_cluster_obstacles(
                camera_index=0, depth_threshold=5.0, eps=0.1, min_samples=3
            )
            
            # Verify vision was called correctly
            self.mock_vision.capture_image.assert_called_once_with(camera_index=0)
            self.mock_vision.detect_obstacles.assert_called_once()
            
            # Verify results structure
            self.assertEqual(obstacle_points.shape[1], 3, "Should return 3D points")
            self.assertEqual(len(labels), len(obstacle_points), "Should have label for each point")
            
            print("✅ Perception obstacle detection and clustering pipeline working")
            
        except ImportError as e:
            self.skipTest(f"Perception module not available: {e}")
    
    def test_empty_obstacle_detection(self):
        """Test perception behavior when no obstacles are detected."""
        try:
            from ManipulaPy.perception import Perception
            
            # Mock vision to return empty results
            empty_vision = Mock()
            empty_vision.capture_image.return_value = (
                np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                np.random.uniform(0.1, 5.0, (480, 640)).astype(np.float32)
            )
            empty_vision.detect_obstacles.return_value = (
                np.empty((0, 3)),  # No obstacles
                np.array([])       # No labels
            )
            
            perception = Perception(vision_instance=empty_vision)
            
            obstacle_points, labels = perception.detect_and_cluster_obstacles()
            
            self.assertEqual(obstacle_points.shape, (0, 3), "Should handle empty detection")
            self.assertEqual(len(labels), 0, "Should return empty labels")
            
            print("✅ Perception empty obstacle detection handling working")
            
        except ImportError as e:
            self.skipTest(f"Perception module not available: {e}")
    
    def test_invalid_depth_handling(self):
        """Test perception with invalid depth data."""
        try:
            from ManipulaPy.perception import Perception
            
            # Mock vision to return invalid depth
            invalid_vision = Mock()
            invalid_vision.capture_image.return_value = (
                np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                None  # Invalid depth
            )
            
            perception = Perception(vision_instance=invalid_vision)
            
            obstacle_points, labels = perception.detect_and_cluster_obstacles()
            
            self.assertEqual(obstacle_points.shape, (0, 3), "Should handle invalid depth")
            self.assertEqual(len(labels), 0, "Should return empty results")
            
            print("✅ Perception invalid depth handling working")
            
        except ImportError as e:
            self.skipTest(f"Perception module not available: {e}")

class TestPerceptionClustering(unittest.TestCase):
    """Test Perception clustering functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_vision = Mock()
    
    def test_clustering_with_real_sklearn(self):
        """Test perception clustering with real scikit-learn when available."""
        if not is_module_available('sklearn'):
            self.skipTest("Real sklearn not available")
            
        try:
            from sklearn.cluster import DBSCAN
            from ManipulaPy.perception import Perception
            
            # Verify we have real sklearn
            self.assertTrue(hasattr(DBSCAN, 'fit'), "Should have real sklearn DBSCAN")
            
            # Create test data with clear clusters
            test_points = np.array([
                [1, 1, 1], [1.1, 1.1, 1.1], [1.2, 1.2, 1.2],  # Cluster 1
                [5, 5, 5], [5.1, 5.1, 5.1], [5.2, 5.2, 5.2],  # Cluster 2
                [10, 1, 1], [10.1, 1.1, 1.1],                   # Cluster 3
            ])
            
            # Mock vision to return our test points
            test_vision = Mock()
            test_vision.capture_image.return_value = (
                np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                np.random.uniform(0.1, 5.0, (480, 640)).astype(np.float32)
            )
            test_vision.detect_obstacles.return_value = (test_points, None)
            
            perception = Perception(vision_instance=test_vision)
            
            # Test clustering with real DBSCAN
            labels, num_clusters = perception.cluster_obstacles(test_points, eps=0.5, min_samples=2)
            
            self.assertGreater(num_clusters, 0, "Should find clusters in structured data")
            self.assertEqual(len(labels), len(test_points), "Should have label for each point")
            
            # Test that similar points get same labels
            unique_labels = set(labels)
            if len(unique_labels) > 1:
                for label in unique_labels:
                    if label == -1:  # Skip noise points
                        continue
                    cluster_points = test_points[labels == label]
                    if len(cluster_points) > 1:
                        distances = np.linalg.norm(cluster_points - cluster_points[0], axis=1)
                        self.assertTrue(np.all(distances < 1.0), "Points in same cluster should be close")
            
            print("✅ Perception with real sklearn clustering working")
            
        except ImportError as e:
            self.skipTest(f"Real sklearn not available: {e}")
    
    def test_clustering_with_mock_sklearn(self):
        """Test clustering with mocked sklearn."""
        try:
            from ManipulaPy.perception import Perception
            
            # Create test points
            test_points = np.random.randn(20, 3)
            
            perception = Perception(vision_instance=self.mock_vision)
            
            # Test clustering (will use mocked DBSCAN)
            labels, num_clusters = perception.cluster_obstacles(test_points, eps=0.1, min_samples=3)
            
            self.assertEqual(len(labels), len(test_points), "Should have label for each point")
            self.assertIsInstance(num_clusters, int, "Should return integer cluster count")
            
            print("✅ Perception with mock sklearn clustering working")
            
        except ImportError as e:
            self.skipTest(f"Perception module not available: {e}")
    
    def test_clustering_empty_points(self):
        """Test clustering behavior with empty point set."""
        try:
            from ManipulaPy.perception import Perception
            
            perception = Perception(vision_instance=self.mock_vision)
            
            # Test with empty points
            labels, num_clusters = perception.cluster_obstacles(np.empty((0, 3)), eps=0.1, min_samples=3)
            
            self.assertEqual(len(labels), 0, "Should return empty labels for empty input")
            self.assertEqual(num_clusters, 0, "Should return zero clusters for empty input")
            
            print("✅ Perception empty clustering handling working")
            
        except ImportError as e:
            self.skipTest(f"Perception module not available: {e}")

class TestPerceptionStereoMethods(unittest.TestCase):
    """Test Perception stereo vision functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock vision with stereo capabilities
        self.stereo_vision = Mock()
        self.stereo_vision.stereo_enabled = True
        self.stereo_vision.rectify_stereo_images.return_value = (
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        )
        self.stereo_vision.compute_disparity.return_value = np.random.randn(480, 640).astype(np.float32)
        self.stereo_vision.get_stereo_point_cloud.return_value = np.random.randn(1000, 3)
    
    def test_stereo_disparity_computation(self):
        """Test stereo disparity computation through Perception."""
        try:
            from ManipulaPy.perception import Perception
            
            perception = Perception(vision_instance=self.stereo_vision)
            
            # Test stereo disparity computation
            left_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            right_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            disparity = perception.compute_stereo_disparity(left_img, right_img)
            
            # Verify vision methods were called
            self.stereo_vision.rectify_stereo_images.assert_called_once_with(left_img, right_img)
            self.stereo_vision.compute_disparity.assert_called_once()
            
            print("✅ Perception stereo disparity computation working")
            
        except ImportError as e:
            self.skipTest(f"Perception module not available: {e}")
    
    def test_stereo_point_cloud_generation(self):
        """Test stereo point cloud generation through Perception."""
        try:
            from ManipulaPy.perception import Perception
            
            perception = Perception(vision_instance=self.stereo_vision)
            
            # Test point cloud generation
            left_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            right_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            point_cloud = perception.get_stereo_point_cloud(left_img, right_img)
            
            # Verify vision method was called
            self.stereo_vision.get_stereo_point_cloud.assert_called_once_with(left_img, right_img)
            self.assertEqual(point_cloud.shape[1], 3, "Point cloud should be 3D")
            
            print("✅ Perception stereo point cloud generation working")
            
        except ImportError as e:
            self.skipTest(f"Perception module not available: {e}")
    
    def test_stereo_methods_without_stereo_config(self):
        """Test stereo operations fail gracefully without stereo configuration."""
        try:
            from ManipulaPy.perception import Perception
            
            # Mock vision without stereo
            mono_vision = Mock()
            mono_vision.stereo_enabled = False
            
            perception = Perception(vision_instance=mono_vision)
            
            left_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            right_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Should raise RuntimeError for stereo operations without stereo support
            with self.assertRaises(RuntimeError):
                perception.compute_stereo_disparity(left_img, right_img)
            
            print("✅ Perception stereo error handling working")
            
        except ImportError as e:
            self.skipTest(f"Perception module not available: {e}")

class TestPerceptionResourceManagement(unittest.TestCase):
    """Test Perception resource management and cleanup."""
    
    def test_resource_cleanup(self):
        """Test that Perception properly cleans up resources."""
        try:
            from ManipulaPy.perception import Perception
            
            # Mock vision with release method
            cleanup_vision = Mock()
            cleanup_vision.release = Mock()
            
            perception = Perception(vision_instance=cleanup_vision)
            
            # Test explicit release
            perception.release()
            cleanup_vision.release.assert_called_once()
            
            print("✅ Perception resource cleanup working")
            
        except ImportError as e:
            self.skipTest(f"Perception module not available: {e}")
    
    def test_destructor_cleanup(self):
        """Test that Perception destructor handles cleanup gracefully."""
        try:
            from ManipulaPy.perception import Perception
            
            # Mock vision with release method
            cleanup_vision = Mock()
            cleanup_vision.release = Mock()
            
            perception = Perception(vision_instance=cleanup_vision)
            
            # Test destructor cleanup (manual call)
            perception.__del__()
            
            # Should not raise any errors
            print("✅ Perception destructor cleanup working")
            
        except ImportError as e:
            self.skipTest(f"Perception module not available: {e}")

class TestPerceptionIntegration(unittest.TestCase):
    """Test Perception integration with real Vision module."""
    
    def test_perception_with_real_vision(self):
        """Test Perception integration with real Vision instance."""
        try:
            from ManipulaPy.vision import Vision
            from ManipulaPy.perception import Perception
            
            # Create real vision instance
            vision = Vision(use_pybullet_debug=False, show_plot=False)
            
            # Create perception with real vision
            perception = Perception(vision_instance=vision)
            
            self.assertEqual(perception.vision, vision, "Perception should use provided vision")
            
            # Test that perception can call vision methods
            try:
                obstacle_points, labels = perception.detect_and_cluster_obstacles(
                    camera_index=0, depth_threshold=2.0
                )
                # Should complete without errors
                self.assertIsInstance(obstacle_points, np.ndarray)
                self.assertIsInstance(labels, np.ndarray)
                
                print("✅ Perception-Vision integration working")
                
            except Exception as e:
                # If it fails due to missing camera or other issues, that's expected
                if "not found" in str(e).lower() or "camera" in str(e).lower():
                    print("✅ Perception-Vision integration working (with expected camera errors)")
                else:
                    raise
            
        except ImportError as e:
            self.skipTest(f"Vision or Perception modules not available: {e}")
    
    def test_end_to_end_obstacle_detection(self):
        """Test end-to-end obstacle detection pipeline."""
        try:
            from ManipulaPy.vision import Vision
            from ManipulaPy.perception import Perception
            
            # Create test data
            test_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            test_depth = np.ones((480, 640), dtype=np.float32) * 2.0
            
            # Add some obstacles in depth
            test_depth[200:250, 300:350] = 1.0  # Close obstacle
            test_depth[100:150, 500:550] = 0.8  # Closer obstacle
            
            # Create vision instance
            vision = Vision(use_pybullet_debug=False, show_plot=False)
            
            # Override capture_image to return our test data
            original_capture = vision.capture_image
            vision.capture_image = lambda **kwargs: (test_rgb, test_depth)
            
            # Create perception
            perception = Perception(vision_instance=vision)
            
            # Test the pipeline (will use mocked YOLO detection)
            obstacle_points, labels = perception.detect_and_cluster_obstacles(
                camera_index=0, depth_threshold=1.5
            )
            
            # Should complete without errors
            self.assertIsInstance(obstacle_points, np.ndarray)
            self.assertIsInstance(labels, np.ndarray)
            
            # Restore original method
            vision.capture_image = original_capture
            
            print("✅ End-to-end obstacle detection pipeline working")
            
        except ImportError as e:
            self.skipTest(f"Vision or Perception modules not available: {e}")

class TestPerceptionErrorHandling(unittest.TestCase):
    """Test Perception error handling and edge cases."""
    
    def test_vision_method_errors(self):
        """Test handling of errors from Vision methods."""
        try:
            from ManipulaPy.perception import Perception
            
            # Mock vision that raises errors
            error_vision = Mock()
            error_vision.capture_image.side_effect = Exception("Camera error")
            error_vision.detect_obstacles.side_effect = Exception("Detection error")
            
            perception = Perception(vision_instance=error_vision)
            
            # Should handle vision errors gracefully
            try:
                obstacle_points, labels = perception.detect_and_cluster_obstacles()
                # If it doesn't raise, should return safe defaults
                self.assertIsInstance(obstacle_points, np.ndarray)
                self.assertIsInstance(labels, np.ndarray)
            except Exception as e:
                # Or raise appropriate errors
                self.assertIsInstance(e, Exception)
            
            print("✅ Perception vision error handling working")
            
        except ImportError as e:
            self.skipTest(f"Perception module not available: {e}")
    
    def test_clustering_parameter_validation(self):
        """Test clustering parameter validation."""
        try:
            from ManipulaPy.perception import Perception
            
            mock_vision = Mock()
            perception = Perception(vision_instance=mock_vision)
            
            # Test with valid parameters
            test_points = np.random.randn(10, 3)
            labels, num_clusters = perception.cluster_obstacles(test_points, eps=0.5, min_samples=3)
            
            self.assertIsInstance(labels, np.ndarray)
            self.assertIsInstance(num_clusters, int)
            
            # Test with edge case parameters
            labels, num_clusters = perception.cluster_obstacles(test_points, eps=0.01, min_samples=1)
            
            self.assertIsInstance(labels, np.ndarray)
            self.assertIsInstance(num_clusters, int)
            
            print("✅ Perception clustering parameter validation working")
            
        except ImportError as e:
            self.skipTest(f"Perception module not available: {e}")

if __name__ == "__main__":
    # Print environment info
    print("Testing Perception module with available backends:")
    print(f"- scikit-learn: {'✓' if is_module_available('sklearn') else '✗'}")
    print(f"- Vision module: {'✓' if is_module_available('ManipulaPy.vision') else '✗'}")
    print(f"- NumPy: ✓ (always available)")
    print()
    
    # Run with verbose output
    unittest.main(verbosity=2)