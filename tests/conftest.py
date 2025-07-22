#!/usr/bin/env python3
"""
Smart conftest.py that only mocks what's truly unavailable or GPU-only.
Allows proper testing of CPU-capable libraries while gracefully handling missing dependencies.
"""

import sys
import os
from unittest.mock import Mock, MagicMock
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
# Suppress warnings during testing
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add the package to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['NUMBA_DISABLE_CUDA'] = '1'
os.environ['MANIPULAPY_FORCE_CPU'] = '1'

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
        # Convert to appropriate dtype, defaulting to float32 for GPU simulation
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
    
    def __str__(self):
        return str(self._data)
    
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
    
    def __isub__(self, other):
        if hasattr(other, '_data'):
            self._data -= other._data.astype(self._data.dtype)
        else:
            self._data -= np.asarray(other, dtype=self._data.dtype)
        return self
    
    def __imul__(self, other):
        if hasattr(other, '_data'):
            self._data *= other._data.astype(self._data.dtype)
        else:
            self._data *= np.asarray(other, dtype=self._data.dtype)
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
    
    # Math functions
    def sqrt(self, arr):
        data = arr._data if hasattr(arr, '_data') else np.asarray(arr)
        return CuPyArrayMock(np.sqrt(data))
    
    def sin(self, arr):
        data = arr._data if hasattr(arr, '_data') else np.asarray(arr)
        return CuPyArrayMock(np.sin(data))
    
    def cos(self, arr):
        data = arr._data if hasattr(arr, '_data') else np.asarray(arr)
        return CuPyArrayMock(np.cos(data))
    
    # Random module
    @property
    def random(self):
        class RandomMock:
            def randn(self, *args, **kwargs):
                return CuPyArrayMock(np.random.randn(*args, **kwargs))
            def uniform(self, low=0, high=1, size=None):
                return CuPyArrayMock(np.random.uniform(low, high, size))
            def normal(self, *args, **kwargs):
                return CuPyArrayMock(np.random.normal(*args, **kwargs))
        return RandomMock()
    
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
            
            def det(self, a):
                a_data = a._data if hasattr(a, '_data') else np.asarray(a)
                return np.linalg.det(a_data)
        
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

def test_module_availability(module_name):
    """Test if a module is available and can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False
    except Exception as e:
        # Module exists but has issues (like missing system deps)
        print(f"Warning: {module_name} has import issues: {e}")
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
        torch_mock.nn = MockModule("torch.nn")
        torch_mock.optim = MockModule("torch.optim")
        torch_mock.device = lambda x: Mock()
        torch_mock.float32 = np.float32
        torch_mock.float64 = np.float64
        return torch_mock
    
    elif module_name == "cv2":
        cv2_mock = MockModule("cv2")
        # OpenCV constants
        cv2_mock.INTER_LINEAR = 1
        cv2_mock.INTER_CUBIC = 2
        cv2_mock.COLOR_BGR2GRAY = 6
        cv2_mock.COLOR_BGR2RGB = 4
        cv2_mock.COLOR_RGB2BGR = 3
        cv2_mock.CV_32FC1 = 5
        cv2_mock.CALIB_ZERO_DISPARITY = 1024
        
        # Mock functions that return reasonable values
        cv2_mock.remap = lambda img, *args, **kwargs: img
        cv2_mock.cvtColor = lambda img, code: img
        cv2_mock.GaussianBlur = lambda img, ksize, sigma: img
        cv2_mock.VideoCapture = lambda x: Mock(isOpened=lambda: False, release=lambda: None)
        
        # Stereo functions
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
        numba_mock.cuda.random = MockModule("numba.cuda.random")
        numba_mock.cuda.random.create_xoroshiro128p_states = lambda n, seed=None: None
        numba_mock.cuda.random.xoroshiro128p_uniform_float32 = lambda states, thread_id: 0.5
        numba_mock.float32 = np.float32
        numba_mock.float64 = np.float64
        numba_mock.int32 = np.int32
        
        return numba_mock
    
    elif module_name == "pybullet":
        pb_mock = MockModule("pybullet")
        
        # Mock PyBullet functions
        pb_mock.connect = lambda mode: 0
        pb_mock.disconnect = lambda: None
        pb_mock.resetSimulation = lambda: None
        pb_mock.setGravity = lambda x, y, z: None
        pb_mock.setTimeStep = lambda dt: None
        pb_mock.loadURDF = lambda path, *args, **kwargs: 0
        pb_mock.getNumJoints = lambda robot_id: 6
        pb_mock.stepSimulation = lambda: None
        
        # Camera functions
        pb_mock.computeViewMatrix = lambda *args, **kwargs: np.eye(4).flatten()
        pb_mock.computeProjectionMatrixFOV = lambda *args, **kwargs: np.eye(4).flatten()
        pb_mock.computeViewMatrixFromYawPitchRoll = lambda *args, **kwargs: np.eye(4).flatten()
        
        def getCameraImage(width, height, **kwargs):
            rgba = np.random.randint(0, 255, (height, width, 4), dtype=np.uint8)
            depth = np.random.uniform(0.1, 5.0, (height, width))
            segmentation = np.zeros((height, width), dtype=np.int32)
            return (width, height, rgba, depth, segmentation)
        
        pb_mock.getCameraImage = getCameraImage
        
        # Joint control
        pb_mock.setJointMotorControlArray = lambda *args, **kwargs: None
        pb_mock.getJointState = lambda robot_id, joint_id: (0.0, 0.0, [0]*6, 0.0)
        pb_mock.getJointInfo = lambda robot_id, joint_id: (
            0, f"joint_{joint_id}", 0, -1, -1, 0, 0.0, 0.0, -np.pi, np.pi, 1000.0, 100.0, 
            f"link_{joint_id}", [0, 0, 1], [0, 0, 0], [0, 0, 0, 1], -1
        )
        
        # Debug interface
        pb_mock.addUserDebugParameter = lambda name, min_val, max_val, default: len(name)
        pb_mock.readUserDebugParameter = lambda param_id: 0.0
        pb_mock.addUserDebugLine = lambda *args, **kwargs: None
        
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
                print(f"Mock URDF visualization for {self}")
            
            def animate(self, *args, **kwargs):
                print("Mock URDF animation")
            
            def link_fk(self, cfg=None):
                return {}
        
        urchin_mock.urdf = MockModule("urchin.urdf")
        urchin_mock.urdf.URDF = MockURDF
        return urchin_mock
    
    else:
        return MockModule(module_name)

# GPU-only modules that should always be mocked
ALWAYS_MOCK = [
    "cupy",              # GPU-only library
    "pycuda",            # GPU-only
    "pycuda.driver",     # GPU-only
    "pycuda.autoinit",   # GPU-only
    "numba.cuda",        # GPU-specific part of numba
    "numba.cuda.random", # GPU-specific
]

# Simulation/complex modules that are problematic in CI but not strictly GPU-only
MOCK_IN_CI = [
    "pybullet",          # Complex simulation, headless issues
    "urchin",            # URDF library, may not be available
    "urchin.urdf",       # URDF library, may not be available
]

# CPU libraries that should be tested when available
TEST_WHEN_AVAILABLE = [
    "torch",             # Has CPU support, should test when available
    "cv2",               # OpenCV - pure CPU library
    "sklearn",           # Scikit-learn - pure CPU library  
    "sklearn.cluster",   # Part of scikit-learn
    "ultralytics",       # YOLO can run on CPU
    "numba",             # Core numba works without CUDA
]

# Setup mocks intelligently
print("🔧 Setting up intelligent test environment...")

# Always mock GPU-only modules
for module_name in ALWAYS_MOCK:
    sys.modules[module_name] = create_smart_mock(module_name)
    print(f"🚫 Always mocked: {module_name} (GPU-only)")

# Mock simulation modules (can be conditional based on CI environment)
in_ci = os.environ.get('CI', '').lower() in ('true', '1', 'yes')
for module_name in MOCK_IN_CI:
    if in_ci or not test_module_availability(module_name):
        sys.modules[module_name] = create_smart_mock(module_name)
        print(f"🤖 Mocked: {module_name} (simulation/CI)")
    else:
        print(f"🎯 Available: {module_name} (will be tested)")

# Only mock CPU libraries if they're actually unavailable
for module_name in TEST_WHEN_AVAILABLE:
    if test_module_availability(module_name):
        print(f"✅ Available: {module_name} (will be tested)")
    else:
        mock = create_smart_mock(module_name)
        if mock is not None:
            sys.modules[module_name] = mock
            print(f"❌ Mocked: {module_name} (unavailable)")

# Set environment variables to ensure CPU-only execution
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TORCH_USE_CUDA_DSA'] = '0'
os.environ['NUMBA_DISABLE_CUDA'] = '1'
os.environ['MANIPULAPY_FORCE_CPU'] = '1'

print("✅ Smart test environment ready - testing real libraries when available!")

# Additional test configuration
import pytest

# Configure pytest to handle async operations if needed
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up the test environment for each test."""
    # Ensure clean state
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    
    yield
    
    # Cleanup after each test
    pass