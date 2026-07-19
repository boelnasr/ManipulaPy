"""Compatibility contracts for the potential-field package decomposition."""

import ast
import hashlib
import inspect
import pickle
import textwrap

import ManipulaPy.potential_field as potential_field
import ManipulaPy.potential_field.fields as fields
from ManipulaPy.potential_field.adjacency import build_link_adjacency
from ManipulaPy.potential_field.collision import CollisionChecker


_BASE_PACKAGE_NAMES = {
    "Any",
    "CollisionChecker",
    "ConvexHull",
    "Dict",
    "Iterable",
    "NDArray",
    "PotentialField",
    "Set",
    "URDF",
    "build_link_adjacency",
    "fields",
    "itertools",
    "logging",
    "np",
}

_BASE_FIELDS_NAMES = {
    "Any",
    "CollisionChecker",
    "ConvexHull",
    "Dict",
    "Iterable",
    "NDArray",
    "PotentialField",
    "Set",
    "URDF",
    "_logger",
    "_to_host_numpy",
    "build_link_adjacency",
    "get_backend",
    "itertools",
    "logging",
    "np",
}

_BASE_CALLABLE_HASHES = {
    "build_link_adjacency": (
        "5e56f1cf61a5e2eafdb3ef44983380457720fa370cdb1abfcc762b8b301a4e4e"
    ),
    "PotentialField.__init__": (
        "b58473399931e1c49c8d2901ba6c170124d0e5ea82a1e27fde641552cc261a0e"
    ),
    "PotentialField.compute_attractive_potential": (
        "4af42468e76f8d2918c73cb8bcaf0f7c370ec1103141f4cb45a18e9b6f055cf5"
    ),
    "PotentialField.compute_repulsive_potential": (
        "387bf7c6c32707e3f65e4811b52c5b4831c55ddc7f966bb01c86008cb3940b8d"
    ),
    "PotentialField.compute_gradient": (
        "787caacc84d40dbba6e60be915ea3e40dc4e91120fea6b868f52eb3a464c02a0"
    ),
    "CollisionChecker.__init__": (
        "0cb51ec4fc06f9700a7dd371e8a3ca316ef9d5a7e009af0a57c98db4f2a80c0e"
    ),
    "CollisionChecker._warn_visual_fallback_once": (
        "575ad9d597f6366730d05cfac9d555a6c1b151b1e9031dc2810386c512648235"
    ),
    "CollisionChecker._create_convex_hulls": (
        "c85d291e94b50ef16f605cdce6c32c4caca9f3ed9d1ac5d91ec74a99c9320397"
    ),
    "CollisionChecker._transform_convex_hull": (
        "90827c5daa134b1d7a28d9b0f34b82d01f548c1e4144d9f7cc779aa8ccee82f0"
    ),
    "CollisionChecker.check_collision": (
        "288a64f0430c3f96e1ca08eebf8e17cc44964c34aa7637735f9d67faf1cb1b67"
    ),
    "CollisionChecker._points_intersect": (
        "b3c416a92047a6c55db4ce97b2109183d15ae18a24bb7f056431ee5132814e8d"
    ),
    "CollisionChecker._hulls_intersect": (
        "c0129eeff2f3b1a898ed08a1920eb0b550daf492a5236fbc88656d897d436052"
    ),
}


def _stable_ast(node):
    """Serialize AST structure without interpreter-specific metadata."""
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


def _callable_hash(function):
    source = textwrap.dedent(inspect.getsource(function))
    node = ast.parse(source).body[0]
    payload = repr(_stable_ast(node)).encode()
    return hashlib.sha256(payload).hexdigest()


def test_s7_modules_own_their_concerns_and_preserve_aliases():
    """The corrected package has real modules without breaking interim aliases."""
    assert potential_field.PotentialField is fields.PotentialField
    assert potential_field.CollisionChecker is CollisionChecker
    assert fields.CollisionChecker is CollisionChecker
    assert potential_field.build_link_adjacency is build_link_adjacency
    assert fields.build_link_adjacency is build_link_adjacency

    assert fields.PotentialField.__module__ == "ManipulaPy.potential_field.fields"
    assert CollisionChecker.__module__ == "ManipulaPy.potential_field.collision"
    assert build_link_adjacency.__module__ == "ManipulaPy.potential_field.adjacency"


def test_s7_preserves_namespace_and_descriptor_surfaces():
    """Only the two intended package submodule attributes are added."""
    package_names = {
        name for name in vars(potential_field) if not name.startswith("__")
    }
    fields_names = {name for name in vars(fields) if not name.startswith("__")}

    assert package_names == _BASE_PACKAGE_NAMES | {"adjacency", "collision"}
    assert fields_names == _BASE_FIELDS_NAMES
    assert {
        name for name in vars(fields.PotentialField) if not name.startswith("__")
    } == {
        "compute_attractive_potential",
        "compute_gradient",
        "compute_repulsive_potential",
    }
    assert {name for name in vars(CollisionChecker) if not name.startswith("__")} == {
        "_create_convex_hulls",
        "_hulls_intersect",
        "_points_intersect",
        "_transform_convex_hull",
        "_warn_visual_fallback_once",
        "check_collision",
    }

    for cls in (fields.PotentialField, CollisionChecker):
        for name, descriptor in vars(cls).items():
            if not name.startswith("__"):
                assert inspect.isfunction(descriptor), name


def test_s7_old_pickle_module_path_still_resolves():
    """Objects pickled through the interim fields alias remain loadable."""
    instance = CollisionChecker.__new__(CollisionChecker)
    restored = pickle.loads(pickle.dumps(instance))

    assert isinstance(restored, CollisionChecker)
    assert isinstance(restored, fields.CollisionChecker)


def test_s7_callable_bodies_match_pre_split_ast_hashes():
    """Every retained or moved callable is structurally identical to pre-split S7."""
    callables = {
        "build_link_adjacency": build_link_adjacency,
        **{
            f"PotentialField.{name}": value
            for name, value in vars(fields.PotentialField).items()
            if inspect.isfunction(value)
        },
        **{
            f"CollisionChecker.{name}": value
            for name, value in vars(CollisionChecker).items()
            if inspect.isfunction(value)
        },
    }
    actual = {name: _callable_hash(value) for name, value in callables.items()}

    assert actual == _BASE_CALLABLE_HASHES
