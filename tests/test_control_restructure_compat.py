"""Compatibility and pure-move guards for the SR8 control split."""

import ast
import hashlib
import importlib
import inspect
import pickle
import textwrap

import pytest

import ManipulaPy.control as control
import ManipulaPy.control.manipulator_controller as implementation


PACKAGE_NAMES = frozenset(
    """
    Any Dict List ManipulatorController NDArray Optional Tuple Union
    _as_backend_array _to_host_array _validate_i_clamp annotations get_backend
    logger logging manipulator_controller np plt
    """.split()
)
IMPLEMENTATION_NAMES = frozenset(
    """
    Any BackendArray Dict List ManipulatorController NDArray Optional _StateOwner
    Tuple Union _as_backend_array _to_host_array _validate_i_clamp annotations
    dataclass get_backend logger logging np plt use_backend
    """.split()
)
METHOD_OWNERS = {
    "computed_torque_control": "computed_torque",
    "feedforward_control": "computed_torque",
    "pd_feedforward_control": "computed_torque",
    "enforce_limits": "computed_torque",
    "joint_space_control": "computed_torque",
    "cartesian_space_control": "computed_torque",
    "pd_control": "pid",
    "pid_control": "pid",
    "robust_control": "robust_adaptive",
    "adaptive_control": "robust_adaptive",
    "kalman_filter_predict": "kalman",
    "kalman_filter_update": "kalman",
    "kalman_filter_control": "kalman",
    "plot_steady_state_response": "metrics",
    "calculate_rise_time": "metrics",
    "calculate_percent_overshoot": "metrics",
    "calculate_settling_time": "metrics",
    "calculate_steady_state_error": "metrics",
    "ziegler_nichols_tuning": "metrics",
    "tune_controller": "metrics",
    "find_ultimate_gain_and_period": "metrics",
}
CLASS_NAMES = frozenset(
    {
        "__dict__",
        "__doc__",
        "__init__",
        "__module__",
        "__weakref__",
        "_normalize_state",
        "_set_state",
        *METHOD_OWNERS,
    }
)
AST_HASHES = {
    "computed_torque_control": "49501b9fdf61cc4cbb699f1e16c8953e663e0dcd64a5b5328f0099f5cae526d8",  # noqa: E501
    "pd_control": "5d19525d4e6f9b66c74624148c0ee84aadd4ca26db9b765aaad9dbf7731bd8c7",
    "pid_control": "ed9c1d933c9bb156c351f8e076652d4d647c0139f0f0c7afdebb5922b66b3cb5",
    "robust_control": "3dd850d4e095f3167acdc124a696f40c623dd72fc7404380bcbd79709a9cb047",  # noqa: E501
    "adaptive_control": "c2ee8da5c8eb1b7becc34967f8067d94eb76cdc10ca7ac1e9b12a61762f4f1b4",  # noqa: E501
    "kalman_filter_predict": "c7b40249d8026cfc9e92d08bda8c1bb5a00d6c31e260f7e06475617bfb31eb94",  # noqa: E501
    "kalman_filter_update": "e29bce5a960fb03b5a4bd6fed7eba14e5434244c65b5c1ffeda7fcd36f3acc27",  # noqa: E501
    "kalman_filter_control": "90d5e71315c2e9084a62a4b5f219411f5ea2825bac896a12fc5cf3889e3dac96",  # noqa: E501
    "feedforward_control": "dedafb51e923949518f103009961f8b459d0b475cdef9863b83af285d221338f",  # noqa: E501
    "pd_feedforward_control": "d0a1d119b6f4a87c363da33e289062d8f00216f639b20eb8174e73fff2269813",  # noqa: E501
    "enforce_limits": "8abe9e424ae89252be7fa428507545208043586dd3b1fb68146fb8ffe88881a8",  # noqa: E501
    "plot_steady_state_response": "4ac8350e0f582eb79b924e7a571badc501fc82e1a8c8235fd1f4168164e823da",  # noqa: E501
    "calculate_rise_time": "8c7f01a1d0f842f5a29210681388867e5846e91ff69bcdb899215cbc60e34043",  # noqa: E501
    "calculate_percent_overshoot": "170f64d478c12694e48e92c0edae2f698c5646ba660afcd9be2db9cc1eae67b0",  # noqa: E501
    "calculate_settling_time": "9c3911311f99842d0fce16306434ecdf7e6337b3974aa932c3616955fc325868",  # noqa: E501
    "calculate_steady_state_error": "aa5f1c304c35f850f2d595eb1ea6eb4d5a143ab6d50a0296ff1e0bbfc7c2e3cf",  # noqa: E501
    "joint_space_control": "17426c4265e1726effa204a5a2e84e0ecebbce60bab4f85cc2bcead6c2ebfb98",  # noqa: E501
    "cartesian_space_control": "06c7214324b1050267aa48c9b7860e63300f6b7ca9baed66890c2313cc4447b1",  # noqa: E501
    "ziegler_nichols_tuning": "3785f71019a8b233d012b9bc0eef95d8049783ed2d8eb143ddfa14632f29bf46",  # noqa: E501
    "tune_controller": "517a36c4bb55bc37a4ba8cfbe174c7665673797c9483b95d1d9a2bb893f065e3",  # noqa: E501
    "find_ultimate_gain_and_period": "2c23306bb54c7f53bc0b359950bc2fca69e2bbc3b3ecf37bd78cc3b822139720",  # noqa: E501
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


def _method_hash(descriptor):
    function = (
        descriptor.__func__ if isinstance(descriptor, staticmethod) else descriptor
    )
    node = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
    node = _RuntimeNormalizer().visit(node)
    return hashlib.sha256(repr(_stable_ast(node)).encode()).hexdigest()


def _names(module):
    return frozenset(name for name in vars(module) if not name.startswith("__"))


def test_sr8_concern_modules_exist():
    for name in ("computed_torque", "pid", "robust_adaptive", "kalman", "metrics"):
        importlib.import_module(f"ManipulaPy.control.{name}")


def test_historical_module_package_and_class_surfaces_are_exact():
    cls = implementation.ManipulatorController
    assert _names(control) == PACKAGE_NAMES
    assert _names(implementation) == IMPLEMENTATION_NAMES
    assert frozenset(cls.__dict__) == CLASS_NAMES
    assert cls.__bases__ == (object,)
    assert cls.__mro__ == (cls, object)


def test_raw_descriptors_and_signatures_are_installed_without_wrappers():
    cls = implementation.ManipulatorController
    for name, module_name in METHOD_OWNERS.items():
        module = importlib.import_module(f"ManipulaPy.control.{module_name}")
        owner = next(
            value
            for key, value in vars(module).items()
            if key.endswith("Concern") and inspect.isclass(value)
        )
        public = inspect.getattr_static(cls, name)
        source = inspect.getattr_static(owner, name)
        assert public is source
        assert type(public) is type(source)
        assert inspect.signature(getattr(cls, name)) == inspect.signature(
            getattr(owner, name)
        )
    assert isinstance(inspect.getattr_static(cls, "enforce_limits"), staticmethod)


def test_all_moved_methods_match_pre_sr8_ast_hashes():
    actual = {
        name: _method_hash(
            inspect.getattr_static(implementation.ManipulatorController, name)
        )
        for name in AST_HASHES
    }
    assert actual == AST_HASHES


def test_public_identity_and_pickle_contracts_are_preserved():
    assert control.ManipulatorController is implementation.ManipulatorController
    instance = implementation.ManipulatorController.__new__(
        implementation.ManipulatorController
    )
    instance.marker = "sr8"
    restored = pickle.loads(pickle.dumps(instance))
    assert type(restored) is implementation.ManipulatorController
    assert restored.marker == "sr8"


def test_package_runtime_rebinding_and_delete_reach_moved_method(monkeypatch):
    original = control._as_backend_array
    calls = []

    def patched(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(control, "_as_backend_array", patched)
    controller = implementation.ManipulatorController.__new__(
        implementation.ManipulatorController
    )
    result = controller.pd_control(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert float(result) == 1.0
    assert len(calls) == 6

    monkeypatch.delattr(control, "_as_backend_array")
    with pytest.raises((AttributeError, NameError)):
        controller.pd_control(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
