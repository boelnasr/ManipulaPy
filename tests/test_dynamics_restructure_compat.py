# flake8: noqa: E501
"""Compatibility contracts for the dynamics package decomposition."""

import ast
import hashlib
import inspect
import pickle
import textwrap

import numpy as np

import ManipulaPy
import ManipulaPy.dynamics as dynamics
import ManipulaPy.dynamics.cache as cache
import ManipulaPy.dynamics.forces as forces
import ManipulaPy.dynamics.id_fd as id_fd
import ManipulaPy.dynamics.manipulator_dynamics as implementation
import ManipulaPy.dynamics.mass_matrix as mass_matrix
from ManipulaPy.kinematics import SerialManipulator


_BASE_IMPLEMENTATION_NAMES = frozenset(
    """
    Any Dict List ManipulatorDynamics NDArray Optional SerialManipulator Tuple
    Union ad get_backend np
    """.split()
)
_RESTRUCTURING_IMPLEMENTATION_NAMES = frozenset(
    {
        "_CacheConcern",
        "_ForcesConcern",
        "_InverseForwardDynamicsConcern",
        "_MassMatrixConcern",
    }
)
_BASE_CLASS_NAMES = frozenset(
    """
    __doc__ __init__ __module__ _concrete_cache_key _gravity_forces_legacy
    _mass_matrix_derivatives _mass_matrix_legacy forward_dynamics gravity_forces
    inverse_dynamics mass_matrix partial_derivative velocity_quadratic_forces
    """.split()
)
_MOVED_METHOD_HASHES = {
    "_concrete_cache_key": "96bd46837d447c74b367698b223917dde1f8edceb9b760ed4b25b8a492388f79",
    "mass_matrix": "a8da8827c2754a286fe44b78ffd0a42a25998551dcfc7043e6cb1db60c2e4b63",
    "_mass_matrix_legacy": "214865b010e6365ee6478be8b6adf4d77b876601957a25bd13c0954a8c0fa83b",
    "_mass_matrix_derivatives": "957fde1c4094b30c4cd61c5113ca05875b9fb2541c42b9cb57e486106811bc25",
    "partial_derivative": "078f7bd17e4d4e9a083a24dc59c167b4bdd9f868874b14056f141c1bb335b898",
    "velocity_quadratic_forces": "a5ddf2956722f91a452944b49fb9bc613bc564bf202ce7b4b7ad4aad171ca177",
    "gravity_forces": "b1ef9b0e1753d49e554ece7bf6a46fc5461eb8ee1ddca5c7860ed5c71080f0fd",
    "_gravity_forces_legacy": "d2d8c22847c3e979d7031e1572ac2640dd1b50bd5139f8a7bbc0c54d2594762b",
    "inverse_dynamics": "cbd56645f4f22cd16218899666ef845d623ccbff79adcf961e65b99d68b302fd",
    "forward_dynamics": "6783a72b2d786f7d48fbac92b259148dde84822608a3b974a5241831f61674ed",
}
_METHOD_OWNERS = {
    "_concrete_cache_key": cache._CacheConcern,
    "_mass_matrix_derivatives": cache._CacheConcern,
    "mass_matrix": mass_matrix._MassMatrixConcern,
    "_mass_matrix_legacy": mass_matrix._MassMatrixConcern,
    "partial_derivative": forces._ForcesConcern,
    "velocity_quadratic_forces": forces._ForcesConcern,
    "gravity_forces": forces._ForcesConcern,
    "_gravity_forces_legacy": forces._ForcesConcern,
    "inverse_dynamics": id_fd._InverseForwardDynamicsConcern,
    "forward_dynamics": id_fd._InverseForwardDynamicsConcern,
}


class _MovedMethodNormalizer(ast.NodeTransformer):
    """Normalize only the dynamic facade lookup required by the pure move."""

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body.pop(0)
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "_runtime":
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        return node


def _stable_ast(node):
    if isinstance(node, ast.AST):
        fields = tuple(
            (name, _stable_ast(value))
            for name, value in ast.iter_fields(node)
            if name not in {"ctx", "type_comment", "type_params"}
        )
        return type(node).__name__, fields
    if isinstance(node, list):
        return tuple(_stable_ast(value) for value in node)
    return node


def _normalized_method_hash(descriptor):
    function = (
        descriptor.__func__ if isinstance(descriptor, staticmethod) else descriptor
    )
    method = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
    normalized = _MovedMethodNormalizer().visit(method)
    return hashlib.sha256(repr(_stable_ast(normalized)).encode()).hexdigest()


def test_public_class_identity_module_and_exact_mro_are_preserved():
    cls = implementation.ManipulatorDynamics
    assert dynamics.ManipulatorDynamics is cls
    assert ManipulaPy.ManipulatorDynamics is cls
    assert cls.__module__ == "ManipulaPy.dynamics.manipulator_dynamics"
    assert cls.__bases__ == (SerialManipulator,)
    assert cls.__mro__ == (cls, *SerialManipulator.__mro__)


def test_facade_and_class_names_are_frozen():
    implementation_names = {n for n in vars(implementation) if not n.startswith("__")}
    assert implementation_names == (
        _BASE_IMPLEMENTATION_NAMES | _RESTRUCTURING_IMPLEMENTATION_NAMES
    )
    assert frozenset(implementation.ManipulatorDynamics.__dict__) == _BASE_CLASS_NAMES


def test_extracted_descriptors_are_installed_without_wrapper_drift():
    cls = implementation.ManipulatorDynamics
    for name, owner in _METHOD_OWNERS.items():
        public_descriptor = inspect.getattr_static(cls, name)
        source_descriptor = inspect.getattr_static(owner, name)
        assert public_descriptor is source_descriptor
        assert type(public_descriptor) is type(source_descriptor)
        assert inspect.signature(getattr(cls, name)) == inspect.signature(
            getattr(owner, name)
        )


def test_every_moved_method_matches_the_pre_split_ast_oracle():
    cls = implementation.ManipulatorDynamics
    actual = {
        name: _normalized_method_hash(inspect.getattr_static(cls, name))
        for name in _MOVED_METHOD_HASHES
    }
    assert actual == _MOVED_METHOD_HASHES


def test_historical_runtime_patch_reaches_extracted_methods(monkeypatch):
    calls = []
    original = implementation.get_backend

    def patched_get_backend():
        calls.append("get_backend")
        return original()

    monkeypatch.setattr(implementation, "get_backend", patched_get_backend)
    instance = implementation.ManipulatorDynamics.__new__(
        implementation.ManipulatorDynamics
    )
    instance._mass_matrix_cache = {}
    instance._mass_matrix_derivative_cache = {}
    instance.mass_matrix = lambda _theta: np.eye(1)
    instance._mass_matrix_derivatives([0.0])
    assert calls == ["get_backend"]
    moved_globals = instance._mass_matrix_derivatives.__globals__
    assert moved_globals["_runtime"] is implementation
    assert "get_backend" not in moved_globals


def test_pickle_round_trip_keeps_historical_class_identity():
    instance = implementation.ManipulatorDynamics.__new__(
        implementation.ManipulatorDynamics
    )
    instance.marker = "s3"
    restored = pickle.loads(pickle.dumps(instance))
    assert type(restored) is implementation.ManipulatorDynamics
    assert restored.marker == "s3"
