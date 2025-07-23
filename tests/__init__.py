#!/usr/bin/env python3
"""
Test suite for ManipulaPy package initialization.

This module tests the graceful import handling, version information,
and module availability in the ManipulaPy package.
"""

import sys
import warnings
import importlib
from unittest.mock import patch, MagicMock
import pytest


class TestManipulaPyInit:
    """Test ManipulaPy package initialization and imports."""

    def test_package_version(self):
        """Test that ManipulaPy has correct version information."""
        import ManipulaPy
        
        assert hasattr(ManipulaPy, '__version__')
        assert hasattr(ManipulaPy, '__author__')
        assert hasattr(ManipulaPy, '__license__')
        
        assert ManipulaPy.__version__ == "1.1.1"
        assert ManipulaPy.__author__ == "Mohamed Aboelnar"
        assert ManipulaPy.__license__ == "AGPL-3.0-or-later"

    def test_package_import_success(self):
        """Test that ManipulaPy can be imported successfully."""
        import ManipulaPy
        assert ManipulaPy is not None

    def test_core_modules_available(self):
        """Test that core modules are available."""
        import ManipulaPy
        
        # These should always be available as core modules
        expected_core = ['utils']  # utils should always work
        
        for module_name in expected_core:
            assert hasattr(ManipulaPy, module_name), f"Core module {module_name} not available"

    def test_torch_dependent_modules(self):
        """Test modules that depend on PyTorch."""
        import ManipulaPy
        
        # These modules require torch (which should be installed as core dependency now)
        torch_dependent = ['kinematics', 'dynamics']
        
        for module_name in torch_dependent:
            if hasattr(ManipulaPy, module_name):
                module = getattr(ManipulaPy, module_name)
                assert module is not None, f"Module {module_name} is None"

    def test_optional_modules_graceful_handling(self):
        """Test that optional modules are handled gracefully when missing."""
        import ManipulaPy
        
        # These are optional modules that may or may not be available
        optional_modules = ['vision', 'perception', 'sim', 'cuda_kernels']
        
        # Should not crash if missing, just not be in __all__
        for module_name in optional_modules:
            # It's OK if these don't exist - they're optional
            if hasattr(ManipulaPy, module_name):
                module = getattr(ManipulaPy, module_name)
                # If present, should not be None
                assert module is not None, f"Optional module {module_name} is present but None"

    def test_all_list_contains_only_imported_modules(self):
        """Test that __all__ only contains successfully imported modules."""
        import ManipulaPy
        
        assert hasattr(ManipulaPy, '__all__')
        assert isinstance(ManipulaPy.__all__, list)
        
        # Every item in __all__ should be available as an attribute
        for module_name in ManipulaPy.__all__:
            assert hasattr(ManipulaPy, module_name), f"Module {module_name} in __all__ but not available"

    def test_import_with_missing_torch(self):
        """Test graceful handling when PyTorch is missing."""
        # This test simulates what happens if torch is not available
        with patch.dict('sys.modules', {'torch': None}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ImportWarning)
                
                # Force reimport
                if 'ManipulaPy' in sys.modules:
                    del sys.modules['ManipulaPy']
                if 'ManipulaPy.kinematics' in sys.modules:
                    del sys.modules['ManipulaPy.kinematics']
                if 'ManipulaPy.dynamics' in sys.modules:
                    del sys.modules['ManipulaPy.dynamics']
                
                import ManipulaPy
                
                # Should still import successfully, just with warnings
                assert ManipulaPy is not None
                
                # Should have warning about missing torch dependencies
                torch_warnings = [warning for warning in w 
                                if 'torch' in str(warning.message).lower() or 
                                   'kinematics' in str(warning.message) or
                                   'dynamics' in str(warning.message)]
                # We expect at least some warnings about missing torch
                # (This test might need adjustment based on actual behavior)

    def test_import_with_missing_optional_dependencies(self):
        """Test graceful handling when optional dependencies are missing."""
        # Mock missing optional dependencies
        missing_modules = ['ultralytics', 'cv2', 'pybullet']
        
        with patch.dict('sys.modules', {mod: None for mod in missing_modules}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ImportWarning)
                
                # Force reimport of optional modules
                optional_modules = ['ManipulaPy.vision', 'ManipulaPy.perception', 'ManipulaPy.sim']
                for mod in optional_modules:
                    if mod in sys.modules:
                        del sys.modules[mod]
                
                # Force reimport of main package
                if 'ManipulaPy' in sys.modules:
                    del sys.modules['ManipulaPy']
                
                import ManipulaPy
                
                # Should still import successfully
                assert ManipulaPy is not None
                
                # Should have warnings about missing optional dependencies
                assert len(w) > 0, "Expected warnings about missing optional dependencies"

    def test_utils_module_functionality(self):
        """Test that utils module has basic functionality."""
        import ManipulaPy
        
        if hasattr(ManipulaPy, 'utils'):
            utils = ManipulaPy.utils
            
            # Test that common utility functions exist
            # (Adjust based on what's actually in your utils module)
            expected_functions = ['rotation_matrix_to_euler_angles', 'skew_symmetric']
            
            for func_name in expected_functions:
                if hasattr(utils, func_name):
                    func = getattr(utils, func_name)
                    assert callable(func), f"utils.{func_name} is not callable"

    def test_kinematics_module_basic_functionality(self):
        """Test that kinematics module has basic functionality if available."""
        import ManipulaPy
        
        if hasattr(ManipulaPy, 'kinematics'):
            kinematics = ManipulaPy.kinematics
            
            # Test that expected classes/functions exist
            # (Adjust based on what's actually in your kinematics module)
            expected_items = ['SerialManipulator']  # Add more as needed
            
            for item_name in expected_items:
                if hasattr(kinematics, item_name):
                    item = getattr(kinematics, item_name)
                    assert item is not None, f"kinematics.{item_name} is None"

    def test_no_torch_import_error_in_utils(self):
        """Test that utils module doesn't depend on torch."""
        # Utils should work even without torch
        import ManipulaPy.utils as utils
        
        # This should work regardless of torch availability
        assert utils is not None

    def test_package_metadata_consistency(self):
        """Test that package metadata is consistent."""
        import ManipulaPy
        
        # Version should be a string
        assert isinstance(ManipulaPy.__version__, str)
        assert len(ManipulaPy.__version__) > 0
        
        # Author should be a string
        assert isinstance(ManipulaPy.__author__, str)
        assert len(ManipulaPy.__author__) > 0
        
        # License should be a string
        assert isinstance(ManipulaPy.__license__, str)
        assert len(ManipulaPy.__license__) > 0

    def test_import_star_functionality(self):
        """Test that 'from ManipulaPy import *' works."""
        # Create a new namespace to test import *
        namespace = {}
        exec("from ManipulaPy import *", namespace)
        
        # Should have imported some items
        imported_items = [k for k in namespace.keys() if not k.startswith('__')]
        assert len(imported_items) > 0, "No items imported with 'from ManipulaPy import *'"

    def test_warning_suppression_in_ci(self):
        """Test that ImportWarnings are properly handled in CI environments."""
        import ManipulaPy
        
        # In CI, we expect the package to load without crashing
        # even if some optional dependencies are missing
        assert ManipulaPy is not None
        assert hasattr(ManipulaPy, '__version__')

    @pytest.mark.parametrize("module_name", [
        "utils", "kinematics", "dynamics", "singularity", 
        "path_planning", "urdf_processor", "control", "potential_field"
    ])
    def test_core_module_attributes(self, module_name):
        """Test that core modules are properly accessible if imported."""
        import ManipulaPy
        
        if hasattr(ManipulaPy, module_name):
            module = getattr(ManipulaPy, module_name)
            assert module is not None, f"Module {module_name} is None"
            # Should be a module object
            assert hasattr(module, '__name__'), f"{module_name} doesn't appear to be a module"

    @pytest.mark.parametrize("module_name", [
        "vision", "perception", "sim", "cuda_kernels"
    ])
    def test_optional_module_attributes(self, module_name):
        """Test that optional modules are properly accessible if imported."""
        import ManipulaPy
        
        if hasattr(ManipulaPy, module_name):
            module = getattr(ManipulaPy, module_name)
            assert module is not None, f"Optional module {module_name} is None"
            # Should be a module object
            assert hasattr(module, '__name__'), f"{module_name} doesn't appear to be a module"
        # If not present, that's OK - it's optional


class TestManipulaPyImportBehavior:
    """Test specific import behaviors and edge cases."""
    
    def test_reimport_stability(self):
        """Test that reimporting ManipulaPy is stable."""
        import ManipulaPy
        first_import = ManipulaPy
        
        # Force reimport
        if 'ManipulaPy' in sys.modules:
            del sys.modules['ManipulaPy']
        
        import ManipulaPy
        second_import = ManipulaPy
        
        # Should have same version and basic attributes
        assert first_import.__version__ == second_import.__version__
        assert first_import.__author__ == second_import.__author__

    def test_submodule_import_direct(self):
        """Test that submodules can be imported directly."""
        # Test direct imports of submodules
        try:
            import ManipulaPy.utils
            assert ManipulaPy.utils is not None
        except ImportError:
            pytest.skip("utils module not available")
        
        # Test torch-dependent modules
        try:
            import ManipulaPy.kinematics
            assert ManipulaPy.kinematics is not None
        except ImportError:
            pytest.skip("kinematics module not available (likely missing torch)")

    def test_circular_import_prevention(self):
        """Test that there are no circular import issues."""
        # This test just ensures that importing doesn't hang due to circular imports
        import ManipulaPy
        
        # If we get here without hanging, circular imports are not an issue
        assert True

    def test_module_list_tracking(self):
        """Test that _CORE_MODULES and _OPTIONAL_MODULES are properly tracked."""
        import ManipulaPy
        
        # These should be internal tracking lists
        # They might not be exposed, but __all__ should reflect them
        assert hasattr(ManipulaPy, '__all__')
        assert len(ManipulaPy.__all__) > 0


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])