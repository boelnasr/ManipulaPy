#!/usr/bin/env python3
"""
Fixed conftest.py with bug fixes for all the test failures.
"""

import sys
import os
from unittest.mock import Mock, MagicMock
import warnings
import numpy as np

# Suppress warnings during testing
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add the package to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

class MockModule:
    """Improved mock module that handles iteration properly."""
    
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
    """Fixed CuPy array mock with proper numpy compatibility."""
    
    def __init__(self, data):
        # Use float64 to avoid casting issues
        self._data = np.asarray(data, dtype=np.float64)
    
    def get(self):
        """CuPy's method to convert GPU array to CPU (numpy) array."""
        return self._data
    
    def __getattr__(self, name):
        return getattr(self._data, name)
    
    def __array__(self):
        return self._data
    
    def __repr__(self):
        return f"CuPyArrayMock({self._data})"
    
    def __str__(self):
        return str(self._data)
    
    # Fixed math operations with proper type handling
    def __add__(self, other):
        if hasattr(other, '_data'):
            result = self._data + other._data
        else:
            result = self._data + np.asarray(other, dtype=self._data.dtype)
        return CuPyArrayMock(result)
    
    def __sub__(self, other):
        if hasattr(other, '_data'):
            result = self._data - other._data
        else:
            result = self._data - np.asarray(other, dtype=self._data.dtype)
        return CuPyArrayMock(result)
    
    def __mul__(self, other):
        if hasattr(other, '_data'):
            result = self._data * other._data
        else:
            result = self._data * np.asarray(other, dtype=self._data.dtype)
        return CuPyArrayMock(result)
    
    def __matmul__(self, other):
        if hasattr(other, '_data'):
            result = self._data @ other._data
        else:
            result = self._data @ np.asarray(other, dtype=self._data.dtype)
        return CuPyArrayMock(result)
    
    def __iadd__(self, other):
        if hasattr(other, '_data'):
            self._data = (self._data + other._data).astype(self._data.dtype)
        else:
            self._data = (self._data + np.asarray(other, dtype=self._data.dtype)).astype(self._data.dtype)
        return self
    
    def __isub__(self, other):
        if hasattr(other, '_data'):
            self._data = (self._data - other._data).astype(self._data.dtype)
        else:
            self._data = (self._data - np.asarray(other, dtype=self._data.dtype)).astype(self._data.dtype)
        return self
    
    def __imul__(self, other):
        if hasattr(other, '_data'):
            self._data = (self._data * other._data).astype(self._data.dtype)
        else:
            self._data = (self._data * np.asarray(other, dtype=self._data.dtype)).astype(self._data.dtype)
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
        return CuPyArrayMock(self._data[key])
    
    def __setitem__(self, key, value):
        if hasattr(value, '_data'):
            self._data[key] = value._data.astype(self._data.dtype)
        else:
            self._data[key] = np.asarray(value, dtype=self._data.dtype)

class CuPyMock:
    """Enhanced CuPy mock that properly handles all CuPy operations."""
    
    def asarray(self, arr):
        if isinstance(arr, CuPyArrayMock):
            return arr
        return CuPyArrayMock(arr)
    
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
    
    # Add random module
    @property
    def random(self):
        class RandomMock:
            def randn(self, *args, **kwargs):
                return CuPyArrayMock(np.random.randn(*args, **kwargs))
            def uniform(self, low=0, high=1, size=None):
                return CuPyArrayMock(np.random.uniform(low, high, size))
        return RandomMock()
    
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

def test_module_availability(module_name):
    """Test if a module is available and can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def create_smart_mock(module_name):
    """Create appropriate mocks only for unavailable modules."""
    
    if module_name == "cupy":
        return CuPyMock()
    
    elif module_name == "torch":
        if test_module_availability("torch"):
            return None
        else:
            torch_mock = MockModule("torch")
            torch_mock.tensor = lambda *args, **kwargs: Mock()
            torch_mock.zeros = lambda *args, **kwargs: Mock()
            torch_mock.cuda = MockModule("torch.cuda")
            torch_mock.cuda.is_available = lambda: False
            torch_mock.nn = MockModule("torch.nn")
            torch_mock.optim = MockModule("torch.optim")
            torch_mock.device = lambda x: Mock()
            torch_mock.float32 = np.float32
            return torch_mock
    
    elif module_name == "cv2":
        if test_module_availability("cv2"):
            return None
        else:
            cv2_mock = MockModule("cv2")
            cv2_mock.INTER_LINEAR = 1
            cv2_mock.COLOR_BGR2GRAY = 6
            cv2_mock.COLOR_BGR2RGB = 4
            cv2_mock.remap = lambda img, *args, **kwargs: img
            cv2_mock.cvtColor = lambda img, code: img
            cv2_mock.VideoCapture = lambda x: Mock(isOpened=lambda: False)
            cv2_mock.StereoSGBM_create = lambda **kwargs: Mock(compute=lambda l, r: np.zeros((480, 640), dtype=np.float32))
            cv2_mock.stereoRectify = lambda *args, **kwargs: (np.eye(3), np.eye(3), np.eye(3, 4), np.eye(3, 4), np.eye(4), None, None)
            cv2_mock.initUndistortRectifyMap = lambda *args, **kwargs: (np.zeros((480, 640), dtype=np.float32), np.zeros((480, 640), dtype=np.float32))
            cv2_mock.reprojectImageTo3D = lambda disp, Q: np.random.randn(disp.shape[0], disp.shape[1], 3)
            cv2_mock.CV_32FC1 = 5
            cv2_mock.GaussianBlur = lambda img, ksize, sigma: img
            return cv2_mock
    
    elif module_name == "sklearn" or module_name == "sklearn.cluster":
        if test_module_availability("sklearn"):
            return None
        else:
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
                        self.labels_ = np.zeros(len(X), dtype=int)
                        if len(X) > 3:
                            self.labels_[len(X)//2:] = 1
                        if len(X) > 6:
                            self.labels_[2*len(X)//3:] = 2
                    return self
            
            sklearn_mock.cluster.DBSCAN = MockDBSCAN
            return sklearn_mock
    
    elif module_name == "ultralytics":
        if test_module_availability("ultralytics"):
            return None
        else:
            ultralytics_mock = MockModule("ultralytics")
            
            class MockYOLO:
                def __init__(self, model_path):
                    self.model_path = model_path
                
                def __call__(self, image, conf=0.3):
                    class MockBoxes:
                        def __init__(self):
                            self.xyxy = [np.array([[100, 100, 200, 200], [300, 300, 400, 400]])]
                        
                        def __len__(self):
                            return len(self.xyxy)
                    
                    class MockResults:
                        def __init__(self):
                            self.boxes = MockBoxes()
                    
                    return [MockResults()]
            
            ultralytics_mock.YOLO = MockYOLO
            return ultralytics_mock
    
    elif module_name == "numba":
        if test_module_availability("numba"):
            return None
        else:
            numba_mock = MockModule("numba")
            
            def jit(*args, **kwargs):
                def decorator(func):
                    return func
                return decorator
            
            numba_mock.jit = jit
            numba_mock.cuda = MockModule("numba.cuda")
            numba_mock.cuda.jit = jit
            numba_mock.cuda.random = MockModule("numba.cuda.random")
            numba_mock.cuda.random.create_xoroshiro128p_states = lambda n, seed=None: None
            numba_mock.cuda.random.xoroshiro128p_uniform_float32 = lambda states, thread_id: 0.5
            numba_mock.float32 = np.float32
            
            return numba_mock
    
    elif module_name == "pybullet":
        # Always mock PyBullet with improved camera support
        pb_mock = MockModule("pybullet")
        
        pb_mock.computeViewMatrix = lambda *args, **kwargs: np.eye(4).flatten()
        pb_mock.computeProjectionMatrixFOV = lambda *args, **kwargs: np.eye(4).flatten()
        
        def getCameraImage(width, height, **kwargs):
            rgba = np.random.randint(0, 255, (height, width, 4), dtype=np.uint8)
            depth = np.random.uniform(0.1, 5.0, (height, width))
            return (
                width,
                height,
                rgba,
                depth,
                np.zeros((height, width))
            )
        
        pb_mock.getCameraImage = getCameraImage
        return pb_mock
    
    else:
        return MockModule(module_name)

# Define modules to check and potentially mock
MODULES_TO_CHECK = [
    "torch",
    "cupy",
    "pycuda",
    "pycuda.driver",
    "pycuda.autoinit",
    "numba",
    "numba.cuda",
    "numba.cuda.random",
    "pybullet",
    "urchin",
    "urchin.urdf",
    "cv2",
    "ultralytics",
    "sklearn",
    "sklearn.cluster",
]

# Always mock these (GPU-only or complex simulation modules)
ALWAYS_MOCK = [
    "cupy",
    "pycuda",
    "pycuda.driver", 
    "pycuda.autoinit",
    "numba.cuda",
    "numba.cuda.random",
    "pybullet",
    "urchin",
    "urchin.urdf",
]

# Setup mocks intelligently
print("Setting up test environment with smart mocking...")

for module_name in MODULES_TO_CHECK:
    if module_name in ALWAYS_MOCK:
        sys.modules[module_name] = create_smart_mock(module_name)
        print(f"Mocked: {module_name} (always mocked)")
    else:
        if test_module_availability(module_name):
            print(f"Available: {module_name} (using real module)")
        else:
            mock = create_smart_mock(module_name)
            if mock is not None:
                sys.modules[module_name] = mock
                print(f"Mocked: {module_name} (unavailable)")

print("Smart mock setup complete - real modules will be tested when available!")

# Set environment variables to force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TORCH_USE_CUDA_DSA'] = '0'
os.environ['NUMBA_DISABLE_CUDA'] = '1'