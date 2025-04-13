# tests/conftest.py
import sys
import types
import os

# Mock third-party modules that are unavailable in CI or test environments
MOCK_MODULES = [
    'torch', 'cupy', 'pycuda', 'pycuda.driver', 'pycuda.autoinit',
    'numba', 'numba.cuda', 'pybullet', 'urchin', 'urchin.urdf',
    'cv2', 'ultralytics', 'sklearn.cluster'
]

class MockModule(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
    def __getattr__(self, attr):
        return MockModule(f"{self.__name__}.{attr}")
    def __call__(self, *args, **kwargs):
        return MockModule(self.__name__ + '.__call__')
    def get(self):
        import numpy as np
        return np.zeros(1)
    def __array__(self):
        import numpy as np
        return np.zeros(1)

for mod_name in MOCK_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MockModule(mod_name)

# Add custom command-line option for skipping CUDA tests
def pytest_addoption(parser):
    parser.addoption(
        "--skip-cuda",
        action="store_true",
        default=False,
        help="Skip tests that require CUDA/GPU execution"
    )

# Configure pytest with environment variable when --skip-cuda is passed
def pytest_configure(config):
    if config.getoption("--skip-cuda"):
        os.environ["SKIP_CUDA_TESTS"] = "true"

