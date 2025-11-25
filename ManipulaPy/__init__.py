#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
ManipulaPy Package

A high-performance robotics library for kinematic and dynamic analysis of serial manipulators.

This package uses lazy loading to minimize import time. Modules are loaded only when accessed:
    - `import ManipulaPy` is fast (<100ms)
    - `from ManipulaPy import SerialManipulator` loads only kinematics
    - Heavy modules (vision, simulation, control) load on demand

License: GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
Copyright (c) 2025 Mohamed Aboelnasr
"""

import sys
import os
from typing import Dict, List, Optional, Any

# Package metadata (always available, no imports)
__version__ = "1.2.0"
__author__  = "Mohamed Aboelnasr"
__license__ = "AGPL-3.0-or-later"

# ---------------------------------------------------------------------
# Lazy Import Infrastructure
# ---------------------------------------------------------------------

# Cache for lazy-loaded modules
_module_cache: Dict[str, Any] = {}

# Track which features are available (computed lazily)
_available_features: Optional[Dict[str, bool]] = None
_missing_dependencies: Dict[str, List[Dict[str, str]]] = {}

def _check_dependency(module_name: str, package_name: str = None, feature: str = None) -> bool:
    """
    Check if a dependency is available without importing it.

    Args:
        module_name: Name of the module to check
        package_name: Name of the package to install (if different from module)
        feature: Feature category this dependency belongs to

    Returns:
        bool: True if dependency is available, False otherwise
    """
    import importlib.util

    spec = importlib.util.find_spec(module_name)
    available = spec is not None

    if not available:
        if package_name is None:
            package_name = module_name

        if feature:
            if feature not in _missing_dependencies:
                _missing_dependencies[feature] = []
            _missing_dependencies[feature].append({
                'module': module_name,
                'package': package_name,
            })

    return available

def _get_available_features() -> Dict[str, bool]:
    """Lazily compute which features are available."""
    global _available_features

    if _available_features is not None:
        return _available_features

    _available_features = {
        'core': True,  # NumPy/SciPy checked on actual import
        'cuda': False,
        'vision': False,
        'simulation': False,
        'ml': False,
    }

    # Check CUDA/GPU support (without importing)
    if _check_dependency('cupy', 'cupy-cuda11x', 'cuda'):
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

    return _available_features

def _lazy_import(module_name: str):
    """
    Lazily import a ManipulaPy submodule.

    Args:
        module_name: Name of the submodule (e.g., 'kinematics', 'control')

    Returns:
        The imported module

    Raises:
        ImportError: If the module cannot be imported
    """
    # Check cache first
    if module_name in _module_cache:
        return _module_cache[module_name]

    # Import the module
    full_name = f"ManipulaPy.{module_name}"
    try:
        import importlib
        module = importlib.import_module(full_name)
        _module_cache[module_name] = module
        return module
    except ImportError as e:
        # Provide helpful error message
        feature_map = {
            'vision': 'vision',
            'perception': 'vision',
            'sim': 'simulation',
            'urdf_processor': 'simulation',
            'cuda_kernels': 'cuda',
        }

        feature = feature_map.get(module_name)
        if feature:
            features = _get_available_features()
            if not features.get(feature, False):
                missing = _missing_dependencies.get(feature, [])
                packages = ', '.join([d['package'] for d in missing])
                raise ImportError(
                    f"ManipulaPy.{module_name} requires {feature} dependencies. "
                    f"Install with: pip install {packages}"
                ) from e

        raise ImportError(f"Failed to import ManipulaPy.{module_name}: {e}") from e

def __getattr__(name: str):
    """
    Lazy module loading via __getattr__.

    When user accesses `ManipulaPy.kinematics`, this function loads it on demand.
    """
    # Map attribute names to module names
    module_map = {
        # Core modules (always available)
        'kinematics': 'kinematics',
        'utils': 'utils',
        'transformations': 'transformations',
        'dynamics': 'dynamics',
        'ik_helpers': 'ik_helpers',

        # Analysis modules
        'control': 'control',
        'path_planning': 'path_planning',
        'singularity': 'singularity',
        'potential_field': 'potential_field',

        # Optional modules
        'vision': 'vision',
        'perception': 'perception',
        'sim': 'sim',
        'urdf_processor': 'urdf_processor',
        'cuda_kernels': 'cuda_kernels',

        # Classes that need to be imported from submodules
        'SerialManipulator': ('kinematics', 'SerialManipulator'),
        'ManipulatorDynamics': ('dynamics', 'ManipulatorDynamics'),
        'ManipulatorController': ('control', 'ManipulatorController'),
        'Vision': ('vision', 'Vision'),
    }

    if name in module_map:
        mapping = module_map[name]

        # Handle direct module imports
        if isinstance(mapping, str):
            return _lazy_import(mapping)

        # Handle class imports (module, class_name)
        elif isinstance(mapping, tuple):
            module_name, class_name = mapping
            module = _lazy_import(module_name)
            return getattr(module, class_name)

    raise AttributeError(f"module 'ManipulaPy' has no attribute '{name}'")

def __dir__():
    """
    Provide tab-completion support for lazy-loaded modules.
    """
    # Core always available
    base = [
        '__version__', '__author__', '__license__',
        'kinematics', 'utils', 'transformations', 'dynamics', 'ik_helpers',
        'control', 'path_planning', 'singularity', 'potential_field',

        # Helper functions
        'check_dependencies', 'get_available_features', 'get_missing_features',
        'require_feature', 'get_installation_command', 'test_installation',

        # Common classes
        'SerialManipulator', 'ManipulatorDynamics', 'ManipulatorController',
    ]

    # Add conditionally available modules
    features = _get_available_features()
    if features.get('vision', False):
        base.extend(['vision', 'perception', 'Vision'])
    if features.get('simulation', False):
        base.extend(['sim', 'urdf_processor'])
    if features.get('cuda', False):
        base.append('cuda_kernels')

    return base

# ---------------------------------------------------------------------
# Helper Functions (lightweight, no heavy imports)
# ---------------------------------------------------------------------

def check_dependencies(verbose: bool = True) -> Dict[str, bool]:
    """
    Check which ManipulaPy features are available.

    Args:
        verbose: If True, print detailed information about missing dependencies

    Returns:
        dict: Dictionary showing which features are available

    Example:
        >>> import ManipulaPy
        >>> ManipulaPy.check_dependencies()
        ManipulaPy Feature Availability Check
        ========================================
        Core        : ✅ Available
        Cuda        : ❌ Not Available
        ...
    """
    features = _get_available_features()

    if verbose:
        print("ManipulaPy Feature Availability Check")
        print("=" * 40)

        for feature, available in features.items():
            status = "✅ Available" if available else "❌ Not Available"
            print(f"{feature.capitalize():<12}: {status}")

            if not available and feature in _missing_dependencies:
                print(f"  Missing dependencies:")
                for dep in _missing_dependencies[feature]:
                    print(f"    - {dep['package']} (pip install {dep['package']})")

        print("\nInstallation commands:")
        if not features['cuda']:
            print("  GPU acceleration: pip install cupy-cuda11x  # or cupy-cuda12x")
        if not features['vision']:
            print("  Vision features:  pip install opencv-python ultralytics pillow")
        if not features['simulation']:
            print("  Simulation:       pip install pybullet")
        if not features['ml']:
            print("  ML features:      pip install torch scikit-learn")

    return features.copy()

def get_available_features() -> List[str]:
    """
    Get list of available feature categories.

    Returns:
        list: Names of available features
    """
    features = _get_available_features()
    return [feature for feature, available in features.items() if available]

def get_missing_features() -> List[str]:
    """
    Get list of missing feature categories.

    Returns:
        list: Names of unavailable features
    """
    features = _get_available_features()
    return [feature for feature, available in features.items() if not available]

def require_feature(feature: str) -> None:
    """
    Raise an error if a required feature is not available.

    Args:
        feature: Feature name to check

    Raises:
        ImportError: If the feature is not available

    Example:
        >>> ManipulaPy.require_feature('vision')
        ImportError: Feature 'vision' not available. Missing dependencies: opencv-python, ...
    """
    features = _get_available_features()

    if feature not in features:
        raise ValueError(f"Unknown feature: {feature}")

    if not features[feature]:
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

def test_installation() -> bool:
    """
    Quick test of ManipulaPy installation and features.

    Returns:
        bool: True if basic functionality works, False otherwise
    """
    print("Testing ManipulaPy Installation...")
    print("-" * 40)

    success = True

    # Test core functionality
    try:
        from ManipulaPy.kinematics import SerialManipulator
        print("✅ Core kinematics: Working")
    except Exception as e:
        print(f"❌ Core kinematics: {e}")
        success = False

    features = _get_available_features()

    # Test vision
    if features.get('vision', False):
        try:
            from ManipulaPy.vision import detect_objects
            print("✅ Vision features: Working")
        except Exception as e:
            print(f"❌ Vision features: {e}")
    else:
        print("⚠️  Vision features: Not available (optional)")

    # Test simulation
    if features.get('simulation', False):
        try:
            import pybullet
            print("✅ Simulation: Working")
        except Exception as e:
            print(f"❌ Simulation: {e}")
    else:
        print("⚠️  Simulation: Not available (optional)")

    # Test GPU
    if features.get('cuda', False):
        try:
            import cupy
            print("✅ GPU acceleration: Available")
        except Exception as e:
            print(f"❌ GPU acceleration: {e}")
    else:
        print("⚠️  GPU acceleration: Not available (optional)")

    print("-" * 40)
    if success:
        print("🎉 ManipulaPy core functionality is working!")
    else:
        print("⚠️  Some issues detected. Check dependencies.")

    return success

def get_version() -> str:
    """Get the ManipulaPy version string."""
    return __version__

# ---------------------------------------------------------------------
# Minimal startup (optional, can be disabled)
# ---------------------------------------------------------------------
if not os.getenv('MANIPULAPY_QUIET', '0') == '1':
    # Lightweight greeting, no heavy imports
    print(f"🤖 ManipulaPy v{__version__} loaded (lazy imports enabled)")
    print("   💡 Use ManipulaPy.check_dependencies() to see available features")
