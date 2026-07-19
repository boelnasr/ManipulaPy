"""Compatibility and pure-move guards for the SR2 kinematics split."""

import ast
import hashlib
import importlib
import inspect
import pickle
import textwrap

import numpy as np

import ManipulaPy
import ManipulaPy.kinematics as kinematics
import ManipulaPy.kinematics.serial_manipulator as implementation


EXPECTED_IMPLEMENTATION_NAMESPACE = {
    "Any",
    "List",
    "NDArray",
    "Optional",
    "SerialManipulator",
    "Tuple",
    "Union",
    "get_backend",
    "np",
    "utils",
}

EXPECTED_PACKAGE_NAMESPACE = {
    "Any",
    "IKInitialGuessCache",
    "List",
    "NDArray",
    "Optional",
    "SerialManipulator",
    "TracIKSolver",
    "Tuple",
    "Union",
    "adaptive_multi_start_ik",
    "extrapolate_from_current",
    "ik_helpers",
    "midpoint_of_limits",
    "np",
    "random_in_limits",
    "serial_manipulator",
    "trac_ik",
    "trac_ik_solve",
    "utils",
    "workspace_heuristic_guess",
}

METHOD_OWNERS = {
    "forward_kinematics": "fk",
    "end_effector_pose": "fk",
    "jacobian": "jacobian",
    "end_effector_velocity": "velocity",
    "joint_velocity": "velocity",
    "iterative_inverse_kinematics": "ik",
    "_pose_error": "ik",
    "smart_inverse_kinematics": "ik",
    "robust_inverse_kinematics": "ik",
    "trac_ik": "ik",
}

EXPECTED_AST_HASHES = {
    "forward_kinematics": (
        "e5a2837f29756621caf6e223b66432d55bdb798d608e717cb092278b1188d4cd"
    ),
    "end_effector_pose": (
        "f244d1266ededa51f3239df6eaca6e2fd5ee491e16cb1cb509b4ca1df6315a26"
    ),
    "jacobian": "f03a1e32f082cb73779c0ee9e22ef242c6042eba12ff06407f38491ca4d990dd",
    "end_effector_velocity": (
        "d25510c9a016fbf7fb99dcb867a03712b9c6a02f68ff19e23c7777ab42926ead"
    ),
    "joint_velocity": (
        "713eaa78e4829cf5c2f359aa89ab33e3a9b498f0bfe5c83490a68113c3da6bea"
    ),
    "iterative_inverse_kinematics": (
        "0311d10380fbaf86b90f4c3e8bccffed9a4363fe7e1e0f42428daf13298f0934"
    ),
    "_pose_error": "209d7ce9c8b5e6e88d6d32c3e875f0b2800e5d2ed7856e21b7e0d64c09d2b4aa",
    "smart_inverse_kinematics": (
        "a12bfcb9389750dba2e136f407e0e954d29628979a6e63c83839fa151498bbb1"
    ),
    "robust_inverse_kinematics": (
        "861cee333ea89aebed86103e49c0c64e59b5bce4682bc658ab04460809426d31"
    ),
    "trac_ik": "07ba10d5968beb1afb41b37dcd44eb9f6ca944c583bbe02cd4cf9d6d8b91be36",
}

EXPECTED_SIGNATURE_HASHES = {
    "forward_kinematics": (
        "aacb52eb4a340e028200dcb814d3689790975d99e29dbd08d83cd925a3db3167"
    ),
    "end_effector_pose": (
        "4647d0f59d33ff768e40557f8f07d61e24819ca7b584da2a98d2c57d9d2f6437"
    ),
    "jacobian": "aacb52eb4a340e028200dcb814d3689790975d99e29dbd08d83cd925a3db3167",
    "end_effector_velocity": (
        "8fa1fcf0680522145773e5bb71083661c61410a12689f0566d814037284d0ddf"
    ),
    "joint_velocity": (
        "2d90d23a7c57465f5367bb94edf6edcc595579dd88ba56a743564cc88b92883d"
    ),
    "iterative_inverse_kinematics": (
        "7eeccd88bfb47bf89fb89c18a691af1ac450bbf4c4780b42eaed1f7ba06a96e0"
    ),
    "_pose_error": "c1d01b7df2c543ed06a71c34be95a45313cac4f1ebfcf6070cbff6f5550ea8fc",
    "smart_inverse_kinematics": (
        "44fe4a0c5b6237c997d270b5ccb7c5dfcda272caeb5cd16a37fc4b81ece9014b"
    ),
    "robust_inverse_kinematics": (
        "d84fb68309b7ff10b1984f5f11ca86fe493f0db896555836c58231afa591565f"
    ),
    "trac_ik": "4590e7a55f9448f49c464af707eb32cf3dcd31a46a5e56ffc38407563171de07",
}

EXPECTED_CLASS_NAMESPACE = {
    "__dict__",
    "__doc__",
    "__init__",
    "__module__",
    "__weakref__",
    "_pose_error",
    "end_effector_pose",
    "end_effector_velocity",
    "forward_kinematics",
    "iterative_inverse_kinematics",
    "jacobian",
    "joint_velocity",
    "robust_inverse_kinematics",
    "smart_inverse_kinematics",
    "trac_ik",
    "update_state",
}


def _public_namespace(module):
    return {name for name in vars(module) if not name.startswith("__")}


class _RuntimeLookupNormalizer(ast.NodeTransformer):
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


def _ast_hash(descriptor):
    function = (
        descriptor.__func__ if isinstance(descriptor, staticmethod) else descriptor
    )
    source = textwrap.dedent(inspect.getsource(function))
    node = ast.parse(source).body[0]
    normalized = _RuntimeLookupNormalizer().visit(node)
    return hashlib.sha256(repr(_stable_ast(normalized)).encode()).hexdigest()


def _signature_hash(function):
    return hashlib.sha256(str(inspect.signature(function)).encode()).hexdigest()


def _robot():
    identity = np.eye(4)
    screw = np.array([[0.0], [0.0], [1.0], [0.0], [0.0], [0.0]])
    return implementation.SerialManipulator(
        identity,
        np.array([[0.0], [0.0], [1.0]]),
        S_list=screw,
        B_list=screw,
        joint_limits=[(-np.pi, np.pi)],
    )


def test_sr2_concern_modules_exist():
    for name in ("fk", "jacobian", "velocity", "ik"):
        importlib.import_module(f"ManipulaPy.kinematics.{name}")


def test_historical_module_and_package_namespaces_are_unchanged():
    assert _public_namespace(implementation) == EXPECTED_IMPLEMENTATION_NAMESPACE
    assert _public_namespace(kinematics) == EXPECTED_PACKAGE_NAMESPACE


def test_serial_manipulator_identity_surface_and_mro_are_preserved():
    cls = implementation.SerialManipulator
    assert kinematics.SerialManipulator is cls
    assert ManipulaPy.SerialManipulator is cls
    assert cls.__bases__ == (object,)
    assert cls.__mro__ == (cls, object)
    assert set(vars(cls)) == EXPECTED_CLASS_NAMESPACE


def test_moved_method_owners_descriptors_and_signatures_are_preserved():
    for method_name, module_name in METHOD_OWNERS.items():
        concern = importlib.import_module(f"ManipulaPy.kinematics.{module_name}")
        owner = next(
            value
            for name, value in vars(concern).items()
            if name.endswith("Concern") and inspect.isclass(value)
        )
        public_descriptor = inspect.getattr_static(
            implementation.SerialManipulator, method_name
        )
        owner_descriptor = inspect.getattr_static(owner, method_name)
        assert public_descriptor is owner_descriptor
        assert type(public_descriptor) is type(owner_descriptor)
        assert _signature_hash(
            getattr(implementation.SerialManipulator, method_name)
        ) == (EXPECTED_SIGNATURE_HASHES[method_name])

    assert isinstance(
        inspect.getattr_static(implementation.SerialManipulator, "_pose_error"),
        staticmethod,
    )


def test_every_moved_method_matches_the_pre_split_ast():
    observed = {
        name: _ast_hash(inspect.getattr_static(implementation.SerialManipulator, name))
        for name in METHOD_OWNERS
    }
    assert observed == EXPECTED_AST_HASHES


def test_serial_manipulator_instances_still_pickle_through_historical_class():
    restored = pickle.loads(pickle.dumps(_robot()))
    assert type(restored) is implementation.SerialManipulator
    np.testing.assert_allclose(restored.S_list, _robot().S_list)


def test_moved_methods_read_the_historical_module_as_canonical_runtime():
    for method_name in METHOD_OWNERS:
        function = getattr(implementation.SerialManipulator, method_name)
        assert function.__globals__["_runtime"] is implementation


def test_runtime_get_backend_supports_set_restore_and_delete():
    robot = _robot()
    original = implementation.get_backend
    calls = []

    def tracking_backend():
        calls.append(True)
        return original()

    try:
        implementation.get_backend = tracking_backend
        robot.forward_kinematics([0.0])
        assert calls == [True]
        implementation.get_backend = original
        robot.forward_kinematics([0.0])
        assert calls == [True]
        del implementation.get_backend
        with np.testing.assert_raises(AttributeError):
            robot.forward_kinematics([0.0])
    finally:
        implementation.get_backend = original


def test_runtime_utils_supports_set_restore_and_delete():
    robot = _robot()
    original = implementation.utils

    class TrackingUtils:
        calls = 0

        def __getattr__(self, name):
            value = getattr(original, name)
            if name == "transform_from_twist":

                def tracked(*args, **kwargs):
                    self.calls += 1
                    return value(*args, **kwargs)

                return tracked
            return value

    tracking = TrackingUtils()
    try:
        implementation.utils = tracking
        robot.forward_kinematics([0.0])
        assert tracking.calls == 1
        implementation.utils = original
        robot.forward_kinematics([0.0])
        assert tracking.calls == 1
        del implementation.utils
        with np.testing.assert_raises(AttributeError):
            robot.forward_kinematics([0.0])
    finally:
        implementation.utils = original


def test_runtime_numpy_supports_set_restore_and_delete():
    robot = _robot()
    original = implementation.np

    class TrackingNumpy:
        calls = 0

        def __getattr__(self, name):
            value = getattr(original, name)
            if name == "concatenate":

                def tracked(*args, **kwargs):
                    self.calls += 1
                    return value(*args, **kwargs)

                return tracked
            return value

    tracking = TrackingNumpy()
    try:
        implementation.np = tracking
        robot.end_effector_pose([0.0])
        assert tracking.calls == 1
        implementation.np = original
        robot.end_effector_pose([0.0])
        assert tracking.calls == 1
        del implementation.np
        with np.testing.assert_raises(AttributeError):
            robot.end_effector_pose([0.0])
    finally:
        implementation.np = original
