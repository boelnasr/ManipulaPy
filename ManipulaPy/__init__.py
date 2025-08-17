#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
ManipulaPy Package

This package provides tools for the analysis and manipulation of robotic systems, including kinematics,
dynamics, singularity analysis, path planning, and URDF processing utilities.

License: GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
Copyright (c) 2025 Mohamed Aboelnar
"""

import warnings
import sys
import os
import platform
from typing import Dict, List, Optional

# Package metadata
__version__ = "1.1.3"            
__author__  = "Mohamed Aboelnar"
__license__ = "AGPL-3.0-or-later"

# Dependency availability tracking
_available_features = {
    'core': True,          # Always available (numpy, scipy, etc.)
    'cuda': False,         # GPU acceleration
    'vision': False,       # Computer vision features
    'simulation': False,   # PyBullet simulation
    'ml': False,          # Machine learning features
}

# Track missing dependencies for helpful error messages
_missing_dependencies = {}

def _check_dependency(module_name: str, package_name: str = None, feature: str = None) -> bool:
    """
    Check if a dependency is available and track missing ones.
    
    Args:
        module_name: Name of the module to import
        package_name: Name of the package to install (if different from module)
        feature: Feature category this dependency belongs to
        
    Returns:
        bool: True if dependency is available, False otherwise
    """
    try:
        __import__(module_name)
        return True
    except ImportError as e:
        if package_name is None:
            package_name = module_name
        
        if feature:
            if feature not in _missing_dependencies:
                _missing_dependencies[feature] = []
            _missing_dependencies[feature].append({
                'module': module_name,
                'package': package_name,
                'error': str(e)
            })
        
        return False

# Check core dependencies (should always be available)
try:
    import numpy as np
    import scipy
    import matplotlib
    _available_features['core'] = True
except ImportError as e:
    _available_features['core'] = False
    raise ImportError(f"Core dependencies missing: {e}. Please reinstall ManipulaPy.")

# Check CUDA/GPU support
if _check_dependency('cupy', 'cupy-cuda11x', 'cuda'):
    _available_features['cuda'] = True
elif _check_dependency('pycuda', 'pycuda', 'cuda'):
    _available_features['cuda'] = True

# Check vision dependencies
vision_deps = [
    ('cv2', 'opencv-python'),
    ('ultralytics', 'ultralytics'),
    ('PIL', 'pillow'),
]
vision_available = all(_check_dependency(mod, pkg, 'vision') for mod, pkg in vision_deps)
_available_features['vision'] = vision_available

# Check simulation dependencies
sim_deps = [
    ('pybullet', 'pybullet'),
    ('urchin', 'urchin'),
    ('trimesh', 'trimesh'),
]
sim_available = all(_check_dependency(mod, pkg, 'simulation') for mod, pkg in sim_deps)
_available_features['simulation'] = sim_available

# Check ML dependencies
ml_deps = [
    ('torch', 'torch'),
    ('sklearn', 'scikit-learn'),
]
ml_available = all(_check_dependency(mod, pkg, 'ml') for mod, pkg in ml_deps)
_available_features['ml'] = ml_available

# ---------------------------------------------------------------------
# Import core modules (always available)
# ---------------------------------------------------------------------
try:
    from ManipulaPy.kinematics import *
    from ManipulaPy.utils import *
    from ManipulaPy.transformations import *
except ImportError as e:
    raise ImportError(f"Failed to import core ManipulaPy modules: {e}")

# ---------------------------------------------------------------------
# Import modules with graceful fallbacks
# ---------------------------------------------------------------------

# Dynamics (core + optional GPU)
try:
    from ManipulaPy.dynamics import *
except ImportError as e:
    warnings.warn(f"Dynamics module not fully available: {e}", UserWarning)

# URDF processing (requires simulation dependencies)
if _available_features['simulation']:
    try:
        from ManipulaPy.urdf_processor import *
    except ImportError as e:
        warnings.warn(f"URDF processing not available: {e}. Install simulation dependencies.", UserWarning)

# Path planning (core + optional GPU)
try:
    from ManipulaPy.path_planning import *
except ImportError as e:
    warnings.warn(f"Path planning not fully available: {e}", UserWarning)

# Control systems (core)
try:
    from ManipulaPy.control import *
except ImportError as e:
    warnings.warn(f"Control systems not available: {e}", UserWarning)

# Singularity analysis (core + optional GPU)
try:
    from ManipulaPy.singularity import *
except ImportError as e:
    warnings.warn(f"Singularity analysis not available: {e}", UserWarning)

# Simulation (requires PyBullet)
if _available_features['simulation']:
    try:
        from ManipulaPy.sim import *
    except ImportError as e:
        warnings.warn(f"Simulation not available: {e}. Install PyBullet: pip install pybullet", UserWarning)

# Potential field (core + optional GPU)
try:
    from ManipulaPy.potential_field import *
except ImportError as e:
    warnings.warn(f"Potential field not available: {e}", UserWarning)

# CUDA kernels (optional)
if _available_features['cuda']:
    try:
        from ManipulaPy.cuda_kernels import *
    except ImportError as e:
        warnings.warn(f"CUDA acceleration not available: {e}. Install cupy: pip install cupy-cuda11x", UserWarning)
        _available_features['cuda'] = False

# Vision and perception (optional)
if _available_features['vision']:
    try:
        from ManipulaPy.vision import *
        from ManipulaPy.perception import *
    except ImportError as e:
        warnings.warn(f"Vision features not available: {e}. Install vision dependencies.", UserWarning)
        _available_features['vision'] = False

# ---------------------------------------------------------------------
# Helper functions for users
# ---------------------------------------------------------------------

def check_dependencies(verbose: bool = True) -> Dict[str, bool]:
    """
    Check which ManipulaPy features are available.
    
    Args:
        verbose: If True, print detailed information about missing dependencies
        
    Returns:
        dict: Dictionary showing which features are available
    """
    if verbose:
        print("ManipulaPy Feature Availability Check")
        print("=" * 40)
        
        for feature, available in _available_features.items():
            status = "✅ Available" if available else "❌ Not Available"
            print(f"{feature.capitalize():<12}: {status}")
            
            if not available and feature in _missing_dependencies:
                print(f"  Missing dependencies:")
                for dep in _missing_dependencies[feature]:
                    print(f"    - {dep['package']} (pip install {dep['package']})")
        
        print("\nInstallation commands:")
        if not _available_features['cuda']:
            print("  GPU acceleration: pip install cupy-cuda11x  # or cupy-cuda12x")
        if not _available_features['vision']:
            print("  Vision features:  pip install opencv-python ultralytics pillow")
            print("  System deps:      sudo apt-get install libgl1-mesa-glx libglib2.0-0  # Ubuntu/Debian")
        if not _available_features['simulation']:
            print("  Simulation:       pip install pybullet urchin trimesh")
        if not _available_features['ml']:
            print("  ML features:      pip install torch scikit-learn")
        
        print(f"\nFor all features: pip install ManipulaPy  # (all deps attempted by default)")
    
    return _available_features.copy()

def get_available_features() -> List[str]:
    """Get list of available feature categories."""
    return [feature for feature, available in _available_features.items() if available]

def get_missing_features() -> List[str]:
    """Get list of missing feature categories."""
    return [feature for feature, available in _available_features.items() if not available]

def require_feature(feature: str) -> None:
    """
    Raise an error if a required feature is not available.
    
    Args:
        feature: Feature name to check
        
    Raises:
        ImportError: If the feature is not available
    """
    if feature not in _available_features:
        raise ValueError(f"Unknown feature: {feature}")
    
    if not _available_features[feature]:
        missing_deps = _missing_dependencies.get(feature, [])
        dep_list = ", ".join([dep['package'] for dep in missing_deps])
        raise ImportError(
            f"Feature '{feature}' not available. Missing dependencies: {dep_list}. "
            f"Install with: pip install {dep_list}"
        )

def get_installation_command(feature: str = None) -> str:
    """
    Get the pip install command for a specific feature or all missing features.
    
    Args:
        feature: Specific feature to get install command for, or None for all missing
        
    Returns:
        str: pip install command
    """
    if feature is not None:
        if feature not in _missing_dependencies:
            return f"# Feature '{feature}' is already available"
        
        deps = [dep['package'] for dep in _missing_dependencies[feature]]
        return f"pip install {' '.join(deps)}"
    
    # Get all missing dependencies
    all_missing = []
    for deps in _missing_dependencies.values():
        for dep in deps:
            if dep['package'] not in all_missing:
                all_missing.append(dep['package'])
    
    if not all_missing:
        return "# All features are already available"
    
    return f"pip install {' '.join(all_missing)}"

def show_feature_status():
    """Print a formatted table of feature availability."""
    try:
        # Try to import tabulate for nice formatting
        from tabulate import tabulate
        
        headers = ["Feature", "Status", "Missing Dependencies"]
        rows = []
        
        for feature, available in _available_features.items():
            status = "✅" if available else "❌"
            missing = ""
            if not available and feature in _missing_dependencies:
                missing_list = [dep['package'] for dep in _missing_dependencies[feature]]
                missing = ", ".join(missing_list)
            
            rows.append([feature.capitalize(), status, missing])
        
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        
    except ImportError:
        # Fallback to simple format if tabulate not available
        check_dependencies(verbose=True)

def print_system_info():
    """Print formatted system information."""
    info = get_system_info()
    print("ManipulaPy System Information")
    print("=" * 35)
    for key, value in info.items():
        print(f"{key.replace('_', ' ').title():<20}: {value}")

def get_system_info() -> Dict[str, str]:
    """Get system information relevant to ManipulaPy."""
    info = {
        'python_version': f"{_sys.version_info.major}.{_sys.version_info.minor}.{_sys.version_info.micro}",
        'platform': _platform.system(),
        'architecture': _platform.machine(),
        'manipulapy_version': __version__,
    }
    
    # Add dependency versions for available features
    if _available_features['core']:
        try:
            import numpy as np
            info['numpy_version'] = np.__version__
        except:
            pass
        try:
            import scipy
            info['scipy_version'] = scipy.__version__
        except:
            pass
        try:
            import matplotlib
            info['matplotlib_version'] = matplotlib.__version__
        except:
            pass
    
    if _available_features['cuda']:
        try:
            import cupy as cp
            info['cupy_version'] = cp.__version__
            info['cuda_version'] = str(cp.cuda.runtime.runtimeGetVersion())
        except:
            pass
    
    if _available_features['vision']:
        try:
            import cv2
            info['opencv_version'] = cv2.__version__
        except:
            pass
        try:
            import ultralytics
            info['ultralytics_version'] = ultralytics.__version__
        except:
            pass
    
    if _available_features['simulation']:
        try:
            import pybullet as p
            info['pybullet_version'] = str(p.getAPIVersion())
        except:
            pass
    
    if _available_features['ml']:
        try:
            import torch
            info['torch_version'] = torch.__version__
        except:
            pass
        try:
            import sklearn
            info['sklearn_version'] = sklearn.__version__
        except:
            pass
    
    return info

# ---------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------

def enable_all_warnings():
    """Enable all ManipulaPy-related warnings for debugging."""
    warnings.filterwarnings("default", category=UserWarning, module="ManipulaPy")

def disable_all_warnings():
    """Disable all ManipulaPy-related warnings."""
    warnings.filterwarnings("ignore", category=UserWarning, module="ManipulaPy")

def test_installation():
    """
    Quick test of ManipulaPy installation and features.
    Returns True if basic functionality works, False otherwise.
    """
    print("Testing ManipulaPy Installation...")
    print("-" * 40)
    
    success = True
    
    # Test core functionality
    try:
        import numpy as np
        from ManipulaPy.kinematics import SerialManipulator
        print("✅ Core kinematics: Working")
    except Exception as e:
        print(f"❌ Core kinematics: {e}")
        success = False
    
    # Test GPU acceleration
    if _available_features['cuda']:
        try:
            from ManipulaPy.cuda_kernels import check_cuda_availability
            if check_cuda_availability():
                print("✅ GPU acceleration: Working")
            else:
                print("⚠️  GPU acceleration: CUDA available but not functional")
        except Exception as e:
            print(f"❌ GPU acceleration: {e}")
    else:
        print("⚠️  GPU acceleration: Not available")
    
    # Test vision
    if _available_features['vision']:
        try:
            import cv2
            print("✅ Vision features: Working")
        except Exception as e:
            print(f"❌ Vision features: {e}")
    else:
        print("⚠️  Vision features: Not available")
    
    # Test simulation
    if _available_features['simulation']:
        try:
            import pybullet as p
            print("✅ Simulation: Working")
        except Exception as e:
            print(f"❌ Simulation: {e}")
    else:
        print("⚠️  Simulation: Not available")
    
    print("-" * 40)
    if success:
        print("🎉 ManipulaPy core functionality is working!")
    else:
        print("⚠️  Some issues detected. Check dependencies.")
    
    return success

# ---------------------------------------------------------------------
# Export control
# ---------------------------------------------------------------------

__all__ = [
    # Core functionality (always available)
    "kinematics",
    "utils", 
    "transformations",
    
    # Feature availability functions
    "check_dependencies",
    "get_available_features",
    "get_missing_features",
    "require_feature",
    "get_installation_command",
    "show_feature_status",
    "get_system_info",
    "print_system_info",
    "test_installation",
    
    # Warning control
    "enable_all_warnings",
    "disable_all_warnings",
    
    # Conditionally available modules
    "dynamics",
    "path_planning", 
    "control",
    "singularity",
]

# Add conditionally available modules to __all__
if _available_features['simulation']:
    __all__.extend(["urdf_processor", "sim"])

if _available_features['cuda']:
    __all__.append("cuda_kernels")

if _available_features['vision']:
    __all__.extend(["vision", "perception"])

__all__.append("potential_field")  # Core + optional GPU

# ---------------------------------------------------------------------
# Startup message (can be disabled with environment variable)
# ---------------------------------------------------------------------
if not os.getenv('MANIPULAPY_QUIET', '0') == '1':
    missing_count = len(get_missing_features())
    if missing_count == 0:
        print(f"🤖 ManipulaPy v{__version__} - All features available!")
    else:
        available_count = len(get_available_features())
        print(f"🤖 ManipulaPy v{__version__} - {available_count} features available, {missing_count} optional features missing")
        print("   Run ManipulaPy.check_dependencies() for details")

# Cleanup namespace (but keep what we need for functions)
_sys = sys
_platform = platform
del warnings, os