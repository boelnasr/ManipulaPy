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
# Computed from the pre-split Simulation methods at commit 8883728 with the
# normalizer above (docstrings included; only ``_runtime.<name>`` reads unwrapped).
AST_HASHES = {
    "set_robot_models": "8d38282263747804a56f4f3171c76e893c9f666f1ced5b6718e3f8eba2d7515d",
    "initialize_planner_and_controller": "f72c9e238ca1b0a36b7a99c88f5bf2719b0fca5dddfc5ee01494da345c084ae2",
    "add_joint_parameters": "f2f823b0ca73d0cc8d8f9dd139cd539e9e2dc99bc9d4fd5f02d9bdaed5056c38",
    "add_reset_button": "d23102d91232c061583586e86aa40bd34f6f13f924cba5fa96c76d1c2cef31f2",
    "set_joint_positions": "5b20965bebb15fe3b9748f1dfc9a2f81af5ddc99dd3f590b49536d4fc668d1e3",
    "get_joint_positions": "a1f679e1dd2ab1f127457aa59f1ff243543b8a797dc90255b0bf9d1239fea698",
    "_capsule_line": "cf21479ee87e554d939067b5b2f3e5ba3d686b51f52f6dc52c8b4a343ad2720b",
    "plot_trajectory": "0f2ecf33eac917bc7ce1b05153fd129687546b56a9f9f07b1b58304951d0228f",
    "_add_trajectory_markers": "f03acc71907fdda6c37e5eb2ae001e6563b02fa0bb0ac5aeddcfc3ffed457ea8",
    "clear_trajectory_visualization": "69cd20579e0a612795128e6cde83150e8cdfb886af9e84203473e934539eac2c",
    "get_joint_parameters": "0758c8f00aa10d74af6e5ab170c840b8d492a1b82a5ff840888aecc29cca2fd0",
    "add_additional_parameters": "9c485580865cc64465dee0a8590364a6fe54b15725291aa92a53022a2ba33fdc",
    "update_simulation_parameters": "8159c1193671409325f0ed4de3075e334152d854e1c239b8635e2a659cb94f9b",
    "save_joint_states": "364d87751984c01593d6d66fe2c5494f67046a9104df58caa81ae0f74f21bf43",
    "plot_trajectory_in_scene": "a9e658c9c129957ba600639238cafd9ed95b3a5f9f3d7669de26e85f1fe5eea7",
}


class _RuntimeNormalizer(ast.NodeTransformer):
    # Docstrings are intentionally NOT stripped: a pure relocation moves the
    # docstring verbatim too, so the hash must guard it. Only the historical
    # ``p``/``pybullet_data``/... reads rewritten to ``_runtime.<name>`` in the
    # moved methods are normalized back, so the hash matches the pre-split body.
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
