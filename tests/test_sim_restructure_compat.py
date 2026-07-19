"""Compatibility and pure-move guards for the SR10 simulation split."""

# ruff: noqa: SIM905 - string snapshots mirror the established SR8 guards.

import ast
import hashlib
import importlib
import inspect
import pickle
import textwrap
import types
from unittest import mock

import pytest

import ManipulaPy.sim as package
import ManipulaPy.sim.simulation as implementation

PACKAGE_NAMES = frozenset(
    """
    Any List ManipulatorController Optional Sequence Simulation Tuple
    _PYBULLET_AVAILABLE logging np os p plt pybullet_data simulation time tp
    """.split()
)
IMPLEMENTATION_NAMES = frozenset(
    """
    Any List ManipulatorController Optional Sequence Simulation Tuple
    _PYBULLET_AVAILABLE _check_pybullet_available get_backend logging np os p plt
    pybullet_data time tp
    """.split()
)
CLASS_NAMES = frozenset(
    """
    __del__ __dict__ __doc__ __init__ __module__ __weakref__
    _add_trajectory_markers _capsule_line _connect_client
    add_additional_parameters add_joint_parameters add_reset_button check_collisions
    clear_trajectory_visualization close_simulation connect_simulation
    disconnect_simulation get_joint_parameters get_joint_positions
    initialize_planner_and_controller initialize_robot manual_control plot_trajectory
    plot_trajectory_in_scene run run_controller run_trajectory save_joint_states
    set_joint_positions set_robot_models setup_logger setup_simulation
    simulate_robot_motion simulate_robot_with_desired_angles step_simulation
    update_simulation_parameters
    """.split()
)
RUNTIME_NAMES = frozenset(
    """
    _FORWARDED_RUNTIME_NAMES _ModuleType _PYBULLET_AVAILABLE
    _SimCompatibilityModule _check_pybullet_available _install_compatibility_proxy
    _sys get_backend p pybullet_data
    """.split()
)
RENDERING_NAMES = frozenset(
    """List Optional Sequence _RenderingConcern _runtime np plt""".split()
)
CONTROLLERS_NAMES = frozenset(
    """
    Any List ManipulatorController Optional Sequence _ControlConcern _runtime np tp
    """.split()
)
METHOD_OWNERS = {
    "_capsule_line": "rendering",
    "plot_trajectory": "rendering",
    "_add_trajectory_markers": "rendering",
    "clear_trajectory_visualization": "rendering",
    "plot_trajectory_in_scene": "rendering",
    "set_robot_models": "controllers",
    "initialize_planner_and_controller": "controllers",
    "add_joint_parameters": "controllers",
    "add_reset_button": "controllers",
    "set_joint_positions": "controllers",
    "get_joint_positions": "controllers",
    "get_joint_parameters": "controllers",
    "add_additional_parameters": "controllers",
    "update_simulation_parameters": "controllers",
    "save_joint_states": "controllers",
}
AST_HASHES = {
    "set_robot_models": "394d5423b656cdcd5bade81acd1174d61b47794522f44c122908f27fc18cde39",
    "initialize_planner_and_controller": "f76b23aae28e4496f2bf66fa5f6122f6db6cefd6f9eabcc269396b60a557b85a",
    "add_joint_parameters": "112feedcb1984635f6532c2cbb6a1f65ed1b2caef512946b0a58da654c17cf94",
    "add_reset_button": "e2a77cce8f2cbbeda5c5b3b53bd849f7ed359639fc42442ab23256db3d85a382",
    "set_joint_positions": "60c4f7ce9985ca9dcaed41c1a14c4e45a5ffd292e31489a2cabcac43ab9d8767",
    "get_joint_positions": "1abfcaf1c4b157edaa74a670b64fd934955fc86b2bf060f52908dbc4afbfbe9b",
    "_capsule_line": "28132ae6f8c8f354ce952f5d4b76742e7ade987c0e0e78b3967a5db3443ddb1f",
    "plot_trajectory": "d9fbe12e336593419b9076f5c1f2d0fc0a99e63af9f699106e69bcfe0c9eedcf",
    "_add_trajectory_markers": "7ce392dec13d9840a185835884bdc44fd784e92ab3f0771096c298623e8cf392",
    "clear_trajectory_visualization": "296d5b0f7d7183fdede450d39b70095159b0a58af7d662897dfbabc193144614",
    "get_joint_parameters": "eb5bf4d488a9fa07a80fa90be24dea68269d0952e3f3d7660f00f9b2bce98895",
    "add_additional_parameters": "2f9aa60e05bd8c30337d45a7dcc2cf43d6db7e2c9f10bb65b18b147aee67d5ad",
    "update_simulation_parameters": "13b400b2bab0a02e40c5eef8ed2f9e599968037cf575f3b16c812b52919cdccc",
    "save_joint_states": "52befa70e5b734f3da9a5dae4f675f3af21b3cde2ad998261c694a2e4a4d311e",
    "plot_trajectory_in_scene": "677344b4df3230b0362fd8d8dd638eace5d797c7bf0f906a09f3a919b5e7054a",
}


class _RuntimeNormalizer(ast.NodeTransformer):
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
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "_runtime"
            and node.attr
            in {
                "p",
                "pybullet_data",
                "_PYBULLET_AVAILABLE",
                "_check_pybullet_available",
                "get_backend",
            }
        ):
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


def _method_hash(descriptor):
    node = ast.parse(textwrap.dedent(inspect.getsource(descriptor))).body[0]
    node = _RuntimeNormalizer().visit(node)
    return hashlib.sha256(repr(_stable_ast(node)).encode()).hexdigest()


def _names(module):
    return frozenset(name for name in vars(module) if not name.startswith("__"))


def _concerns():
    rendering = importlib.import_module("ManipulaPy.sim.rendering")
    controllers = importlib.import_module("ManipulaPy.sim.controllers")
    return rendering, controllers


def test_sr10_concern_modules_exist():
    rendering, controllers = _concerns()
    assert _names(rendering) == RENDERING_NAMES
    assert _names(controllers) == CONTROLLERS_NAMES
    runtime = importlib.import_module("ManipulaPy.sim._runtime")
    assert _names(runtime) == RUNTIME_NAMES


def test_historical_module_package_and_class_surfaces_are_exact():
    cls = implementation.Simulation
    assert _names(package) == PACKAGE_NAMES
    assert _names(implementation) == IMPLEMENTATION_NAMES
    assert frozenset(cls.__dict__) == CLASS_NAMES
    assert cls.__bases__ == (object,)
    assert cls.__mro__ == (cls, object)
    assert cls.__name__ == "Simulation"
    assert cls.__module__ == "ManipulaPy.sim.simulation"
    for name in CLASS_NAMES - {"__dict__", "__doc__", "__module__", "__weakref__"}:
        assert isinstance(inspect.getattr_static(cls, name), types.FunctionType)


def test_raw_descriptors_and_signatures_are_installed_without_wrappers():
    rendering, controllers = _concerns()
    owners = {
        "rendering": rendering._RenderingConcern,
        "controllers": controllers._ControlConcern,
    }
    for name, owner_name in METHOD_OWNERS.items():
        public = inspect.getattr_static(implementation.Simulation, name)
        source = inspect.getattr_static(owners[owner_name], name)
        assert public is source
        assert type(public) is type(source)
        assert inspect.signature(getattr(implementation.Simulation, name)) == (
            inspect.signature(getattr(owners[owner_name], name))
        )


def test_all_moved_methods_match_pre_sr10_ast_hashes():
    actual = {
        name: _method_hash(inspect.getattr_static(implementation.Simulation, name))
        for name in AST_HASHES
    }
    assert actual == AST_HASHES


def test_public_identity_reexports_and_pickle_contracts_are_preserved():
    imported_package = importlib.import_module("ManipulaPy.sim")
    assert package.Simulation is implementation.Simulation
    assert imported_package.Simulation is implementation.Simulation
    assert implementation.Simulation.__module__ == "ManipulaPy.sim.simulation"
    for name in PACKAGE_NAMES - {"simulation"}:
        assert getattr(package, name) is getattr(implementation, name)

    instance = implementation.Simulation.__new__(implementation.Simulation)
    instance.marker = "sr10"
    restored = pickle.loads(pickle.dumps(instance))
    assert type(restored) is implementation.Simulation
    assert restored.marker == "sr10"


def test_runtime_rebinding_and_deletion_synchronize_both_views():
    runtime = importlib.import_module("ManipulaPy.sim._runtime")
    for name in (
        "p",
        "pybullet_data",
        "_PYBULLET_AVAILABLE",
        "_check_pybullet_available",
        "get_backend",
    ):
        original = getattr(implementation, name)
        sentinel = object()
        setattr(implementation, name, sentinel)
        assert implementation.__dict__[name] is sentinel
        assert runtime.__dict__[name] is sentinel
        assert getattr(implementation, name) is sentinel

        delattr(implementation, name)
        assert name not in implementation.__dict__
        assert name not in runtime.__dict__
        with pytest.raises(AttributeError):
            getattr(implementation, name)

        setattr(implementation, name, original)
        assert getattr(implementation, name) is original
        assert getattr(runtime, name) is original

    original_p = implementation.p
    patched_p = object()
    with mock.patch.object(implementation, "p", patched_p):
        assert implementation.p is patched_p
        assert runtime.p is patched_p
    assert implementation.p is original_p
    assert runtime.p is original_p


def test_runtime_rebinding_reaches_staying_and_moved_method_execution(monkeypatch):
    calls = []

    class FakePybullet:
        GUI = "GUI"
        DIRECT = "DIRECT"
        POSITION_CONTROL = "POSITION_CONTROL"

        def connect(self, mode):
            calls.append(("connect", mode))
            return 17

        def resetSimulation(self):
            calls.append(("reset",))

        def setAdditionalSearchPath(self, path):
            calls.append(("search", path))

        def setGravity(self, *gravity):
            calls.append(("gravity", gravity))

        def setTimeStep(self, time_step):
            calls.append(("time_step", time_step))

        def setJointMotorControlArray(self, *args, **kwargs):
            calls.append(("joint", args, kwargs))

    fake = FakePybullet()
    monkeypatch.setattr(implementation, "p", fake)
    monkeypatch.setattr(
        implementation,
        "pybullet_data",
        types.SimpleNamespace(getDataPath=lambda: "/fake/data"),
    )
    monkeypatch.setattr(implementation, "_PYBULLET_AVAILABLE", True)

    simulation = implementation.Simulation.__new__(implementation.Simulation)
    simulation.logger = types.SimpleNamespace(info=lambda *_args: None)
    simulation.physics_client = None
    simulation.time_step = 0.01
    simulation.robot_id = 3
    simulation.non_fixed_joints = [1]
    simulation.connect_simulation()
    simulation.set_joint_positions([0.25])
    assert calls == [
        ("connect", "GUI"),
        ("reset",),
        ("search", "/fake/data"),
        ("gravity", (0, 0, -9.81)),
        ("time_step", 0.01),
        (
            "joint",
            (3, [1], "POSITION_CONTROL"),
            {"targetPositions": [0.25], "forces": [1000.0]},
        ),
    ]
