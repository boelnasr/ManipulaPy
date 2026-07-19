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
    "forward_kinematics": "a47e68e699a71a2fd97544a3c2bb0bd43c8de56b6498c2c7ec1ec87ddd9db0aa",
    "end_effector_velocity": "9402a78b0854ecbc82e903f84ae3c2738b77cb51d4271633c51b4c257d68af95",
    "jacobian": "6781717e551830feee52d20f39e753c5daa0caa6bb58c97ca7724adc267ea513",
    "iterative_inverse_kinematics": "a610e4a4fe4c07338042567ea7a8b234df47ace3be2430b9c18a3d5993f35397",
    "_pose_error": "e8fecd5a664232bbbd637b1c35fe981f48817349a695d303229be7583a75c155",
    "smart_inverse_kinematics": "a7601685fe0d396b7188b869db8662b2b4ffb1e8826f1bb629c62e2d4442e881",
    "robust_inverse_kinematics": "3b950fa3257e09eaed133397ce234d904098fa7ff7c2cb85defd29f2f12b79a2",
    "joint_velocity": "ecc5e76b0691b49fdba63b317de9b2f4330e0d18e33cb7371180ad44171f0a6e",
    "end_effector_pose": "0b571d59f6af1fb299571f84f5fca218ef246983b7b760c2fdc8f78ce16e3c9e",
    "trac_ik": "128f0a598a6b280271777edba47b1f1c5977266fbde8a02a74b9f22e2dcc9d01",
}

EXPECTED_PARAMETERS = {
    "forward_kinematics": ("self", "thetalist", "frame"),
    "end_effector_pose": ("self", "thetalist"),
    "jacobian": ("self", "thetalist", "frame"),
    "end_effector_velocity": ("self", "thetalist", "dthetalist", "frame"),
    "joint_velocity": ("self", "thetalist", "V_ee", "frame"),
    "iterative_inverse_kinematics": (
        "self",
        "T_desired",
        "thetalist0",
        "eomg",
        "ev",
        "max_iterations",
        "plot_residuals",
        "damping",
        "step_cap",
        "png_name",
        "weight_orientation",
        "weight_position",
        "adaptive_tuning",
        "backtracking",
    ),
    "_pose_error": ("T_curr", "T_desired"),
    "smart_inverse_kinematics": (
        "self",
        "T_desired",
        "strategy",
        "theta_current",
        "T_current",
        "cache",
        "eomg",
        "ev",
        "max_iterations",
        "plot_residuals",
        "damping",
        "step_cap",
        "png_name",
        "weight_orientation",
        "weight_position",
        "adaptive_tuning",
        "backtracking",
        "auto_fallback",
    ),
    "robust_inverse_kinematics": (
        "self",
        "T_desired",
        "max_attempts",
        "eomg",
        "ev",
        "max_iterations",
        "verbose",
    ),
    "trac_ik": (
        "self",
        "T_desired",
        "theta0",
        "timeout",
        "eomg",
        "ev",
        "num_restarts",
        "use_parallel",
    ),
}


def _public_namespace(module):
    return {name for name in vars(module) if not name.startswith("__")}


def _ast_hash(function):
    source = textwrap.dedent(inspect.getsource(function))
    node = ast.parse(source).body[0]
    payload = ast.dump(node, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(payload.encode()).hexdigest()


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
    assert [base.__name__ for base in cls.__mro__] == [
        "SerialManipulator",
        "_ForwardKinematicsMixin",
        "_JacobianMixin",
        "_VelocityMixin",
        "_InverseKinematicsMixin",
        "object",
    ]
    assert {name for name in dir(cls) if not name.startswith("__")} == set(
        METHOD_OWNERS
    ) | {"update_state"}


def test_moved_method_owners_descriptors_and_signatures_are_preserved():
    for method_name, module_name in METHOD_OWNERS.items():
        concern = importlib.import_module(f"ManipulaPy.kinematics.{module_name}")
        mixin = next(
            value
            for name, value in vars(concern).items()
            if name.endswith("Mixin") and inspect.isclass(value)
        )
        assert getattr(implementation.SerialManipulator, method_name) is getattr(
            mixin, method_name
        )
        assert (
            tuple(
                inspect.signature(
                    getattr(implementation.SerialManipulator, method_name)
                ).parameters
            )
            == EXPECTED_PARAMETERS[method_name]
        )

    assert isinstance(
        inspect.getattr_static(implementation.SerialManipulator, "_pose_error"),
        staticmethod,
    )


def test_every_moved_method_matches_the_pre_split_ast():
    observed = {
        name: _ast_hash(getattr(implementation.SerialManipulator, name))
        for name in METHOD_OWNERS
    }
    assert observed == EXPECTED_AST_HASHES


def test_serial_manipulator_instances_still_pickle_through_historical_class():
    restored = pickle.loads(pickle.dumps(_robot()))
    assert type(restored) is implementation.SerialManipulator
    np.testing.assert_allclose(restored.S_list, _robot().S_list)


def test_historical_get_backend_rebinding_reaches_moved_methods(monkeypatch):
    original = implementation.get_backend
    calls = []

    def tracking_backend():
        calls.append(True)
        return original()

    monkeypatch.setattr(implementation, "get_backend", tracking_backend)
    robot = _robot()
    robot.forward_kinematics([0.0])
    robot.jacobian([0.0])
    robot.end_effector_velocity([0.0], [0.0])
    implementation.SerialManipulator._pose_error(np.eye(4), np.eye(4))
    assert len(calls) == 5
