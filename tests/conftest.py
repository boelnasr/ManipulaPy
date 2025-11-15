#!/usr/bin/env python3
"""
Smart conftest.py for ManipulaPy test suite.

This configuration:
- Only mocks what's truly unavailable or GPU-only
- Allows proper testing of CPU-capable libraries when available
- Provides comprehensive test fixtures and utilities
- Handles CI environments gracefully
- Enables proper coverage reporting

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import sys
import os
from unittest.mock import Mock, MagicMock
import warnings
import numpy as np
import pytest
import types

# Set up matplotlib for headless testing
import matplotlib
matplotlib.use('Agg')

# Suppress common warnings during testing
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

# Add the package to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Force CPU-only execution
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['NUMBA_DISABLE_CUDA'] = '1'
os.environ['MANIPULAPY_FORCE_CPU'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '0'
# Files/dirs pytest should skip during collection
collect_ignore = [
    "setup.py",
    "ManipulaPy_data",
    "build",
    "dist",
    "test_error_computation.py",
    "test_cuda_kernels.py",  # GPU-only tests
]

# IK convergence tests in parent directory (not in tests/)
collect_ignore_glob = []


# ============================================================================
# Enhanced Mock Classes
# ============================================================================

class MockModule:
    """Enhanced mock module that handles iteration and common operations properly."""
    
    def __init__(self, name=None):
        self._name = name or "MockModule"
        
    def __getattr__(self, name):
        return MockModule(f"{self._name}.{name}")
    
    def __call__(self, *args, **kwargs):
        return MockModule(f"{self._name}()")
    
    def __iter__(self):
        return iter([])
    
    def __bool__(self):
        return True
    
    def __len__(self):
        return 0
    
    def __getitem__(self, key):
        return MockModule(f"{self._name}[{key}]")

class CuPyArrayMock:
    """Enhanced CuPy array mock with proper numpy compatibility and type safety."""
    
    def __init__(self, data, dtype=None):
        if dtype is None:
            dtype = np.float32 if np.issubdtype(np.asarray(data).dtype, np.floating) else np.asarray(data).dtype
        self._data = np.asarray(data, dtype=dtype)
    
    def get(self):
        """CuPy's method to convert GPU array to CPU (numpy) array."""
        return self._data
    
    def copy_to_host(self):
        """Alternative CuPy method for GPU->CPU transfer."""
        return self._data.copy()
    
    def copy_to_device(self, src):
        """Mock GPU memory copy."""
        self._data[:] = np.asarray(src, dtype=self._data.dtype)
    
    def __getattr__(self, name):
        return getattr(self._data, name)
    
    def __array__(self):
        return self._data
    
    def __repr__(self):
        return f"CuPyArrayMock({self._data})"
    
    # Math operations with proper type handling
    def __add__(self, other):
        if hasattr(other, '_data'):
            result = self._data + other._data
        else:
            result = self._data + np.asarray(other, dtype=self._data.dtype)
        return CuPyArrayMock(result, dtype=self._data.dtype)
    
    def __sub__(self, other):
        if hasattr(other, '_data'):
            result = self._data - other._data
        else:
            result = self._data - np.asarray(other, dtype=self._data.dtype)
        return CuPyArrayMock(result, dtype=self._data.dtype)
    
    def __mul__(self, other):
        if hasattr(other, '_data'):
            result = self._data * other._data
        else:
            result = self._data * np.asarray(other, dtype=self._data.dtype)
        return CuPyArrayMock(result, dtype=self._data.dtype)
    
    def __matmul__(self, other):
        if hasattr(other, '_data'):
            result = self._data @ other._data
        else:
            result = self._data @ np.asarray(other, dtype=self._data.dtype)
        return CuPyArrayMock(result, dtype=self._data.dtype)
    
    def __iadd__(self, other):
        if hasattr(other, '_data'):
            self._data += other._data.astype(self._data.dtype)
        else:
            self._data += np.asarray(other, dtype=self._data.dtype)
        return self
    
    # Array properties
    @property
    def shape(self):
        return self._data.shape
    
    @property
    def dtype(self):
        return self._data.dtype
    
    @property
    def size(self):
        return self._data.size
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, key):
        return CuPyArrayMock(self._data[key], dtype=self._data.dtype)
    
    def __setitem__(self, key, value):
        if hasattr(value, '_data'):
            self._data[key] = value._data.astype(self._data.dtype)
        else:
            self._data[key] = np.asarray(value, dtype=self._data.dtype)

class CuPyMock:
    """Comprehensive CuPy mock that properly handles all CuPy operations."""
    
    def asarray(self, arr, dtype=None):
        if isinstance(arr, CuPyArrayMock):
            if dtype is not None and arr.dtype != dtype:
                return CuPyArrayMock(arr._data, dtype=dtype)
            return arr
        return CuPyArrayMock(arr, dtype=dtype)
    
    def asnumpy(self, arr):
        if hasattr(arr, 'get'):
            return arr.get()
        elif hasattr(arr, '_data'):
            return arr._data
        else:
            return np.asarray(arr)
    
    def zeros(self, *args, **kwargs):
        return CuPyArrayMock(np.zeros(*args, **kwargs))
    
    def ones(self, *args, **kwargs):
        return CuPyArrayMock(np.ones(*args, **kwargs))
    
    def zeros_like(self, arr):
        if hasattr(arr, '_data'):
            return CuPyArrayMock(np.zeros_like(arr._data))
        return CuPyArrayMock(np.zeros_like(arr))
    
    def eye(self, n, **kwargs):
        return CuPyArrayMock(np.eye(n, **kwargs))
    
    def array(self, arr, **kwargs):
        return CuPyArrayMock(np.array(arr, **kwargs))
    
    def concatenate(self, arrays, **kwargs):
        numpy_arrays = []
        for arr in arrays:
            if hasattr(arr, '_data'):
                numpy_arrays.append(arr._data)
            else:
                numpy_arrays.append(np.asarray(arr))
        return CuPyArrayMock(np.concatenate(numpy_arrays, **kwargs))
    
    def clip(self, arr, a_min, a_max):
        if hasattr(arr, '_data'):
            data = arr._data
        else:
            data = np.asarray(arr)
        
        if hasattr(a_min, '_data'):
            a_min = a_min._data
        if hasattr(a_max, '_data'):
            a_max = a_max._data
            
        return CuPyArrayMock(np.clip(data, a_min, a_max))
    
    # Linear algebra
    @property
    def linalg(self):
        class LinAlg:
            def solve(self, a, b):
                a_data = a._data if hasattr(a, '_data') else np.asarray(a)
                b_data = b._data if hasattr(b, '_data') else np.asarray(b)
                return CuPyArrayMock(np.linalg.solve(a_data, b_data))
            
            def inv(self, a):
                a_data = a._data if hasattr(a, '_data') else np.asarray(a)
                return CuPyArrayMock(np.linalg.inv(a_data))
            
            def norm(self, a, **kwargs):
                a_data = a._data if hasattr(a, '_data') else np.asarray(a)
                result = np.linalg.norm(a_data, **kwargs)
                if np.isscalar(result):
                    return result
                return CuPyArrayMock(result)
        
        return LinAlg()
    
    def __getattr__(self, name):
        numpy_func = getattr(np, name, None)
        if numpy_func is not None:
            def wrapped_func(*args, **kwargs):
                new_args = []
                for arg in args:
                    if hasattr(arg, '_data'):
                        new_args.append(arg._data)
                    else:
                        new_args.append(arg)
                
                result = numpy_func(*new_args, **kwargs)
                
                if isinstance(result, np.ndarray):
                    return CuPyArrayMock(result)
                else:
                    return result
            
            return wrapped_func
        
        return MockModule(f"cupy.{name}")

# ============================================================================
# Dependency Testing and Smart Mocking
# ============================================================================

def test_module_availability(module_name):
    """Test if a module is available and can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False
    except Exception as e:
        # Module exists but has issues (like missing system deps)
        return False

def create_smart_mock(module_name):
    """Create appropriate mocks for unavailable modules."""
    
    if module_name == "cupy":
        return CuPyMock()
    
    elif module_name == "torch":
        torch_mock = MockModule("torch")
        torch_mock.tensor = lambda *args, **kwargs: Mock()
        torch_mock.zeros = lambda *args, **kwargs: Mock()
        torch_mock.ones = lambda *args, **kwargs: Mock()
        torch_mock.eye = lambda *args, **kwargs: Mock()
        torch_mock.cuda = MockModule("torch.cuda")
        torch_mock.cuda.is_available = lambda: False
        torch_mock.device = lambda x: Mock()
        torch_mock.float32 = np.float32
        torch_mock.float64 = np.float64
        return torch_mock
    
    elif module_name == "cv2":
        cv2_mock = MockModule("cv2")
        # OpenCV constants
        cv2_mock.INTER_LINEAR = 1
        cv2_mock.COLOR_BGR2GRAY = 6
        cv2_mock.CV_32FC1 = 5
        cv2_mock.CALIB_ZERO_DISPARITY = 1024
        
        # Mock functions
        cv2_mock.remap = lambda img, *args, **kwargs: img
        cv2_mock.cvtColor = lambda img, code: img
        cv2_mock.VideoCapture = lambda x: Mock(isOpened=lambda: False)
        cv2_mock.StereoSGBM_create = lambda **kwargs: Mock(
            compute=lambda l, r: np.zeros((480, 640), dtype=np.float32)
        )
        cv2_mock.stereoRectify = lambda *args, **kwargs: (
            np.eye(3), np.eye(3), np.eye(3, 4), np.eye(3, 4), np.eye(4), None, None
        )
        cv2_mock.initUndistortRectifyMap = lambda *args, **kwargs: (
            np.zeros((480, 640), dtype=np.float32), 
            np.zeros((480, 640), dtype=np.float32)
        )
        cv2_mock.reprojectImageTo3D = lambda disp, Q: np.random.randn(disp.shape[0], disp.shape[1], 3)
        
        return cv2_mock
    
    elif module_name in ["sklearn", "sklearn.cluster"]:
        sklearn_mock = MockModule("sklearn")
        sklearn_mock.cluster = MockModule("sklearn.cluster")
        
        class MockDBSCAN:
            def __init__(self, eps=0.5, min_samples=5):
                self.eps = eps
                self.min_samples = min_samples
                self.labels_ = None
            
            def fit(self, X):
                if len(X) == 0:
                    self.labels_ = np.array([])
                else:
                    # Generate reasonable clustering labels
                    self.labels_ = np.zeros(len(X), dtype=int)
                    if len(X) > 3:
                        self.labels_[len(X)//2:] = 1
                    if len(X) > 6:
                        self.labels_[2*len(X)//3:] = 2
                    # Add some noise points
                    if len(X) > 10:
                        self.labels_[::7] = -1
                return self
        
        sklearn_mock.cluster.DBSCAN = MockDBSCAN
        return sklearn_mock
    
    elif module_name == "ultralytics":
        ultralytics_mock = MockModule("ultralytics")
        
        class MockYOLO:
            def __init__(self, model_path):
                self.model_path = model_path
            
            def __call__(self, image, conf=0.3):
                class MockBoxes:
                    def __init__(self):
                        # Generate some reasonable bounding boxes
                        h, w = image.shape[:2] if hasattr(image, 'shape') else (480, 640)
                        self.xyxy = [np.array([
                            [w*0.1, h*0.1, w*0.3, h*0.4],  # Object 1
                            [w*0.6, h*0.5, w*0.9, h*0.8],  # Object 2
                        ])]
                    
                    def __len__(self):
                        return len(self.xyxy[0])
                    
                    def __iter__(self):
                        for box in self.xyxy[0]:
                            yield Mock(xyxy=[box])
                
                class MockResults:
                    def __init__(self):
                        self.boxes = MockBoxes()
                
                return [MockResults()]
        
        ultralytics_mock.YOLO = MockYOLO
        return ultralytics_mock
    
    elif module_name == "numba":
        numba_mock = MockModule("numba")
        
        def jit(*args, **kwargs):
            def decorator(func):
                return func
            return decorator if args else jit
        
        def njit(*args, **kwargs):
            def decorator(func):
                return func
            return decorator if args else njit
        
        def prange(*args, **kwargs):
            return range(*args, **kwargs)
        
        numba_mock.jit = jit
        numba_mock.njit = njit
        numba_mock.prange = prange
        numba_mock.cuda = MockModule("numba.cuda")
        numba_mock.cuda.jit = jit
        numba_mock.float32 = np.float32
        numba_mock.int32 = np.int32
        
        return numba_mock
    
    elif module_name == "pybullet":
        pb_mock = MockModule("pybullet")
        
        # Mock PyBullet functions
        pb_mock.connect = lambda mode: 0
        pb_mock.disconnect = lambda: None
        pb_mock.resetSimulation = lambda: None
        pb_mock.setGravity = lambda x, y, z: None
        pb_mock.loadURDF = lambda path, *args, **kwargs: 0
        pb_mock.getNumJoints = lambda robot_id: 6
        pb_mock.stepSimulation = lambda: None
        
        def getCameraImage(width, height, **kwargs):
            rgba = np.random.randint(0, 255, (height, width, 4), dtype=np.uint8)
            depth = np.random.uniform(0.1, 5.0, (height, width))
            segmentation = np.zeros((height, width), dtype=np.int32)
            return (width, height, rgba, depth, segmentation)
        
        pb_mock.getCameraImage = getCameraImage
        pb_mock.setJointMotorControlArray = lambda *args, **kwargs: None
        pb_mock.getJointState = lambda robot_id, joint_id: (0.0, 0.0, [0]*6, 0.0)
        pb_mock.addUserDebugParameter = lambda name, min_val, max_val, default: len(name)
        pb_mock.readUserDebugParameter = lambda param_id: 0.0
        
        # Constants
        pb_mock.GUI = 1
        pb_mock.DIRECT = 2
        pb_mock.POSITION_CONTROL = 1
        pb_mock.JOINT_REVOLUTE = 0
        pb_mock.JOINT_FIXED = 4
        
        return pb_mock
    
    elif module_name in ["urchin", "urchin.urdf"]:
        urchin_mock = MockModule("urchin")
        
        class MockURDF:
            def __init__(self):
                self.links = []
                self.joints = []
                self.actuated_joints = []
            
            @staticmethod
            def load(urdf_path):
                return MockURDF()
            
            def show(self):
                pass
            
            def animate(self, *args, **kwargs):
                pass
            
            def link_fk(self, cfg=None):
                return {}
        
        urchin_mock.urdf = MockModule("urchin.urdf")
        urchin_mock.urdf.URDF = MockURDF
        return urchin_mock
    
    else:
        return MockModule(module_name)

# ============================================================================
# Intelligent Mocking Setup
# ============================================================================

# GPU-only modules that should always be mocked
ALWAYS_MOCK = [
    "cupy", "pycuda", "pycuda.driver", "pycuda.autoinit",
    "numba.cuda", "numba.cuda.random",
    "torchvision", "torchvision.ops", "torchvision.transforms", "torchvision.io",
]

# Simulation/complex modules that are problematic in CI
MOCK_IN_CI = [
    "pybullet",
    "urchin",
    "urchin.urdf",
]

# CPU libraries that should be tested when available
TEST_WHEN_AVAILABLE = [
    "torch",
    "cv2", 
    "sklearn",
    "sklearn.cluster",
    "ultralytics",
    "numba",
]

print("🔧 Setting up intelligent test environment...")

# Always mock GPU-only modules
for module_name in ALWAYS_MOCK:
    sys.modules[module_name] = create_smart_mock(module_name)

# Mock simulation modules in CI
in_ci = os.environ.get('CI', '').lower() in ('true', '1', 'yes')
for module_name in MOCK_IN_CI:
    if in_ci or not test_module_availability(module_name):
        sys.modules[module_name] = create_smart_mock(module_name)

# Only mock CPU libraries if they're actually unavailable
for module_name in TEST_WHEN_AVAILABLE:
    if not test_module_availability(module_name):
        mock = create_smart_mock(module_name)
        if mock is not None:
            sys.modules[module_name] = mock

# ============================================================================
# Dependency Availability Checks for Test Markers
# ============================================================================

CUDA_AVAILABLE = False  # Always false due to mocking
CUPY_AVAILABLE = False  # Always false due to mocking
OPENCV_AVAILABLE = test_module_availability('cv2')
PYBULLET_AVAILABLE = test_module_availability('pybullet') and not in_ci
YOLO_AVAILABLE = test_module_availability('ultralytics')
SKLEARN_AVAILABLE = test_module_availability('sklearn')
TORCH_AVAILABLE = test_module_availability('torch')

# Environment flags from CI
SKIP_CUDA_TESTS = True  # Always skip due to mocking
SKIP_VISION_TESTS = os.environ.get('SKIP_VISION_TESTS', 'false').lower() == 'true'
SKIP_SIMULATION_TESTS = os.environ.get('SKIP_SIMULATION_TESTS', 'false').lower() == 'true' or in_ci

# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--skip-cuda",
        action="store_true",
        default=True,  # Always skip CUDA
        help="Skip tests that require CUDA/GPU"
    )
    parser.addoption(
        "--skip-vision",
        action="store_true", 
        default=SKIP_VISION_TESTS,
        help="Skip tests that require vision dependencies"
    )
    parser.addoption(
        "--skip-simulation",
        action="store_true",
        default=SKIP_SIMULATION_TESTS,
        help="Skip tests that require simulation dependencies"
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )

def pytest_configure(config):
    """Configure pytest markers and environment."""
    # Register custom markers
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA")
    config.addinivalue_line("markers", "gpu: alias for cuda marker")
    config.addinivalue_line("markers", "vision: mark test as requiring vision deps")
    config.addinivalue_line("markers", "simulation: mark test as requiring simulation deps")
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "convergence: mark test as IK convergence test")
    config.addinivalue_line("markers", "kinematics: mark test as kinematics-related")
    config.addinivalue_line("markers", "dynamics: mark test as dynamics-related")
    config.addinivalue_line("markers", "control: mark test as control-related")
    config.addinivalue_line("markers", "path_planning: mark test as path planning-related")
    config.addinivalue_line("markers", "potential_field: mark test as potential field-related")
    config.addinivalue_line("markers", "singularity: mark test as singularity-related")

def pytest_collection_modifyitems(config, items):
    """Modify test collection based on available dependencies and options."""
    # Skip markers
    skip_cuda = pytest.mark.skip(reason="CUDA not available (always skipped in tests)")
    skip_vision = pytest.mark.skip(reason="Vision dependencies not available or skipped")
    skip_simulation = pytest.mark.skip(reason="Simulation dependencies not available or skipped")
    skip_slow = pytest.mark.skip(reason="Slow tests skipped (use --run-slow to run)")
    
    for item in items:
        # Skip CUDA tests (always)
        if "cuda" in item.keywords or "gpu" in item.keywords:
            item.add_marker(skip_cuda)
        
        # Skip vision tests if not available
        if "vision" in item.keywords and (not OPENCV_AVAILABLE or config.getoption("--skip-vision")):
            item.add_marker(skip_vision)
            
        # Skip simulation tests if not available  
        if "simulation" in item.keywords and (not PYBULLET_AVAILABLE or config.getoption("--skip-simulation")):
            item.add_marker(skip_simulation)
            
        # Skip slow tests unless requested
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)

# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def tolerance():
    """Default numerical tolerance for tests."""
    return 1e-6

@pytest.fixture
def simple_robot_config():
    """Create a simple robot configuration for testing."""
    num_joints = 6
    
    # Home configuration matrix
    M_list = np.eye(4, dtype=np.float32)
    M_list[:3, 3] = [0, 0, 1.0]  # End-effector at (0,0,1)
    
    # Screw axes for a simple 6-DOF robot
    S_list = np.zeros((6, num_joints), dtype=np.float32)
    for i in range(num_joints):
        if i % 2 == 0:  # Revolute joints around Z
            S_list[2, i] = 1.0
        else:  # Revolute joints around Y  
            S_list[1, i] = 1.0
        S_list[5, i] = 0.1 * i  # Some translation component
    
    # Body screw axes (simplified)
    B_list = S_list.copy()
    
    # Inertia matrices
    G_list = []
    for i in range(num_joints):
        G = np.eye(6, dtype=np.float32)
        G[:3, :3] *= 0.1  # Inertia tensor
        G[3:, 3:] *= 1.0  # Mass
        G_list.append(G)
    
    # Joint limits
    joint_limits = [(-np.pi, np.pi) for _ in range(num_joints)]
    torque_limits = [(-100.0, 100.0) for _ in range(num_joints)]
    
    return {
        'M_list': M_list,
        'S_list': S_list,
        'B_list': B_list, 
        'G_list': G_list,
        'joint_limits': joint_limits,
        'torque_limits': torque_limits,
        'num_joints': num_joints
    }

@pytest.fixture
def sample_joint_angles():
    """Sample joint angles for testing."""
    return np.array([0.1, 0.2, -0.3, 0.1, 0.2, 0.1], dtype=np.float32)

@pytest.fixture
def planar_2link_robot():
    """Create a simple 2-link planar robot for IK convergence testing."""
    # Simple 2-link planar robot configuration
    M_list = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1.0],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    S_list = np.array([
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, -1.0, 0]
    ], dtype=np.float64).T

    B_list = np.array([
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, -0.5, 0]
    ], dtype=np.float64).T

    omega_list = [[0, 0, 1], [0, 0, 1]]
    r_list = [[0, 0, 0.5], [0, 0, 1.0]]
    b_list = [[0, 0, 0], [0, 0, 0]]

    return {
        'M_list': M_list,
        'S_list': S_list,
        'B_list': B_list,
        'omega_list': omega_list,
        'r_list': r_list,
        'b_list': b_list,
        'num_joints': 2
    }

@pytest.fixture
def ik_test_angles():
    """Standard test angles for IK convergence testing."""
    return [
        [0.0, 0.0],
        [0.1, 0.1],
        [0.5, 0.3],
        [np.pi/6, np.pi/4],
        [np.pi/4, np.pi/6],
        [1.0, 0.5],
        [-0.5, 0.8],
        [0.7, -0.3],
    ]

@pytest.fixture
def ik_default_params():
    """Default IK parameters for convergence testing."""
    return {
        'eomg': 1e-3,
        'ev': 1e-3,
        'max_iterations': 200,
        'damping': 0.01,
        'step_cap': 0.1
    }

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up the test environment for each test."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Suppress additional warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    yield
    
    # Cleanup after each test
    pass

# ============================================================================
# Utility Functions
# ============================================================================

def assert_array_almost_equal(actual, expected, tolerance=1e-6, msg=""):
    """Assert that two arrays are almost equal within tolerance."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    
    assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} != {expected.shape}. {msg}"
    
    diff = np.abs(actual - expected)
    max_diff = np.max(diff)
    
    assert max_diff < tolerance, (
        f"Arrays not equal within tolerance {tolerance}. "
        f"Max difference: {max_diff}. {msg}"
    )

def requires_dependency(dependency_name):
    """Decorator to skip tests if dependency is not available."""
    availability_map = {
        'cuda': False,  # Always false
        'cupy': False,  # Always false
        'opencv': OPENCV_AVAILABLE and not SKIP_VISION_TESTS,
        'pybullet': PYBULLET_AVAILABLE and not SKIP_SIMULATION_TESTS,
        'yolo': YOLO_AVAILABLE and not SKIP_VISION_TESTS,
        'sklearn': SKLEARN_AVAILABLE,
        'torch': TORCH_AVAILABLE,
    }

    available = availability_map.get(dependency_name.lower(), False)

    return pytest.mark.skipif(
        not available,
        reason=f"{dependency_name} not available or skipped"
    )

def run_ik_convergence_test(robot, target_angles, initial_guess, params=None):
    """
    Helper function to run IK convergence tests with standard parameters.

    Args:
        robot: SerialManipulator instance
        target_angles: Target joint angles
        initial_guess: Initial guess for IK
        params: Dict with IK parameters (eomg, ev, max_iterations, damping, step_cap)

    Returns:
        dict: Results with keys 'result', 'success', 'iterations', 'error'
    """
    if params is None:
        params = {
            'eomg': 1e-3,
            'ev': 1e-3,
            'max_iterations': 200,
            'damping': 0.01,
            'step_cap': 0.1
        }

    target = robot.forward_kinematics(target_angles)
    result, success, iters = robot.iterative_inverse_kinematics(
        target,
        initial_guess,
        eomg=params.get('eomg', 1e-3),
        ev=params.get('ev', 1e-3),
        max_iterations=params.get('max_iterations', 200),
        damping=params.get('damping', 0.01),
        step_cap=params.get('step_cap', 0.1)
    )

    error = np.linalg.norm(np.array(result) - np.array(target_angles))

    return {
        'result': result,
        'success': success,
        'iterations': iters,
        'error': error,
        'target_angles': target_angles,
        'initial_guess': initial_guess
    }

def print_convergence_summary(results_list):
    """
    Print a formatted summary of convergence test results.

    Args:
        results_list: List of result dicts from run_ik_convergence_test
    """
    successes = sum(1 for r in results_list if r['success'])
    total = len(results_list)
    avg_iters = sum(r['iterations'] for r in results_list) / total if total > 0 else 0

    print("\n" + "=" * 60)
    print(f"Convergence rate: {successes}/{total} ({100*successes/total:.1f}%)")
    print(f"Average iterations: {avg_iters:.1f}")
    print("=" * 60)

    for i, result in enumerate(results_list, 1):
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"Test {i}: {status} - {result['iterations']} iterations")
        if not result['success']:
            print(f"  Error: {result['error']:.4f} rad")

    return successes / total if total > 0 else 0

# ============================================================================
# Session Reporting
# ============================================================================

def pytest_sessionstart(session):
    """Report test session start with environment info."""
    print("\n" + "="*60)
    print("ManipulaPy Test Suite")
    print("="*60)
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    
    # Dependency status
    deps = [
        ("CUDA/Numba", False),  # Always false due to mocking
        ("CuPy", False),       # Always false due to mocking
        ("OpenCV", OPENCV_AVAILABLE and not SKIP_VISION_TESTS),
        ("PyBullet", PYBULLET_AVAILABLE and not SKIP_SIMULATION_TESTS),
        ("YOLO", YOLO_AVAILABLE and not SKIP_VISION_TESTS),
        ("Scikit-learn", SKLEARN_AVAILABLE),
        ("PyTorch", TORCH_AVAILABLE),
    ]
    
    print("\nDependency Status:")
    for name, available in deps:
        status = "✅" if available else "❌"
        print(f"  {status} {name}")
    
    print("="*60)

print("✅ Smart test environment ready - testing real libraries when available!")

# ============================================================================
# Available Test Files Documentation
# ============================================================================
"""
ManipulaPy Test Suite - Available Tests:

Core Module Tests (in tests/):
  - test_utils.py                   : Core utility functions (45 tests)
  - test_kinematics.py             : Forward/inverse kinematics
  - test_dynamics.py               : Robot dynamics and equations of motion
  - test_control.py                : Control algorithms (PID, computed torque, etc.)
  - test_control_unit.py           : Unit tests for control module (7 tests)
  - test_trajectory_planning.py    : Trajectory generation and planning
  - test_singularity.py            : Basic singularity analysis
  - test_singularity_extended.py   : Extended singularity tests (24 tests)
  - test_potential_field.py        : Potential field path planning (9 tests)
  - test_potential_field_extended.py: Extended potential field tests (27 tests)
  - test_path_planning_cpu.py      : CPU-based path planning
  - test_path_planning_unit.py     : Unit tests for path planning

Vision & Perception Tests:
  - test_vision.py                 : Vision processing and camera handling
  - test_perception.py             : Object detection and 3D perception

Simulation Tests:
  - test_sim.py                    : PyBullet simulation tests
  - test_urdf_processor.py         : URDF file processing

GPU/CUDA Tests (always skipped in test environment):
  - test_cuda_kernels.py           : GPU kernel tests (skipped)
  - test_cuda_kernels_cpu.py       : CPU fallback for CUDA kernels

Smoke Tests:
  - test_smoke.py                  : Basic import and functionality smoke tests

IK Convergence Tests (in parent directory):
  - test_ik_quick.py              : Quick convergence test with random init
  - test_ik_zero_init.py          : Convergence test with zero initial guess
  - test_ik_diagnostic.py         : Parameter tuning diagnostic tests
  - test_ik_fix.py                : IK fix verification tests

Total Test Count:
  - Core tests: ~96 tests (utils + potential_field + singularity)
  - Control tests: 15 tests (7 unit + 8 integration)
  - All tests: 150+ tests across all modules

Test Markers:
  @pytest.mark.cuda           : Requires CUDA/GPU (always skipped)
  @pytest.mark.gpu            : Alias for cuda marker
  @pytest.mark.vision         : Requires OpenCV/vision dependencies
  @pytest.mark.simulation     : Requires PyBullet
  @pytest.mark.slow           : Slow-running tests (use --run-slow)
  @pytest.mark.integration    : Integration tests
  @pytest.mark.unit           : Unit tests
  @pytest.mark.convergence    : IK convergence tests
  @pytest.mark.kinematics     : Kinematics-related tests
  @pytest.mark.dynamics       : Dynamics-related tests
  @pytest.mark.control        : Control-related tests
  @pytest.mark.path_planning  : Path planning tests
  @pytest.mark.potential_field: Potential field tests
  @pytest.mark.singularity    : Singularity analysis tests

Running Tests:
  # All tests
  pytest tests/

  # Specific module
  pytest tests/test_utils.py

  # By marker
  pytest -m unit
  pytest -m control
  pytest -m convergence

  # Quick core tests
  pytest tests/test_utils.py tests/test_potential_field_extended.py tests/test_singularity_extended.py

  # With coverage
  pytest --cov=ManipulaPy --cov-report=term-missing

  # Specific test
  pytest tests/test_control_unit.py::test_pid_control_zero_gains -v
"""
