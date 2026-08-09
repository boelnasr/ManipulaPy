"""Deterministic Panda calculations for advanced documentation motion studies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ManipulaPy.ManipulaPy_data import get_robot_urdf
from ManipulaPy.dynamics import ManipulatorDynamics
from ManipulaPy.kinematics import SerialManipulator
from ManipulaPy.singularity import Singularity
from ManipulaPy.urdf.types import JointType
from ManipulaPy.urdf_processor import URDFToSerialManipulator


STUDY_TIME = np.linspace(0.0, 4.0, 61, dtype=np.float64)
START = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.785], dtype=np.float64)
MID = np.array([0.15, -0.45, 0.1, -1.8, 0.05, 1.45, 0.65], dtype=np.float64)
GOAL = np.array(
    [-0.35, 0.25, 0.3, -1.55, -0.2, 1.25, -0.4], dtype=np.float64
)
NEAR_SINGULAR = np.array(
    [
        -1.14373813,
        -0.2547399,
        1.3507626,
        -0.42578919,
        1.59465886,
        1.99271463,
        -2.05067442,
    ],
    dtype=np.float64,
)
OBSTACLE_Q = 0.5 * (START + GOAL)
GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float64)
TOOL_WRENCH = np.zeros(6, dtype=np.float64)
JOINT_CLEARANCE = 0.20
SINGULARITY_THRESHOLD = 1e-4


@dataclass(frozen=True)
class PandaFixture:
    """Public Panda models and seven-arm-joint metadata used by every study."""

    serial: SerialManipulator
    dynamics: ManipulatorDynamics
    urdf_path: Path
    joint_names: tuple[str, ...]
    joint_limits: tuple[tuple[float, float], ...]


def load_panda_fixture() -> PandaFixture:
    """Load the bundled Panda through the built-in, mesh-free URDF path."""
    urdf_path = Path(get_robot_urdf("panda"))
    processor = URDFToSerialManipulator(
        urdf_path, backend="builtin", load_meshes=False
    )
    joints = tuple(
        joint
        for joint in processor.robot.actuated_joints
        if joint.joint_type is JointType.REVOLUTE
        and joint.name.startswith("panda_joint")
    )
    if len(joints) != 7 or any(joint.limit is None for joint in joints):
        raise RuntimeError("Panda motion studies require seven limited arm joints")
    return PandaFixture(
        serial=processor.serial_manipulator,
        dynamics=processor.dynamics,
        urdf_path=urdf_path,
        joint_names=tuple(joint.name for joint in joints),
        joint_limits=tuple(
            (float(joint.limit.lower), float(joint.limit.upper)) for joint in joints
        ),
    )


def assert_finite(name: str, value: Any) -> NDArray[np.float64]:
    """Return a host float64 array or reject an invalid scientific result."""
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise RuntimeError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class DynamicsResults:
    """Time histories used by the three dynamics motion studies."""

    time: NDArray[np.float64]
    theta: NDArray[np.float64]
    velocity: NDArray[np.float64]
    acceleration: NDArray[np.float64]
    mass_matrices: NDArray[np.float64]
    inertia: NDArray[np.float64]
    velocity_force: NDArray[np.float64]
    gravity: NDArray[np.float64]
    tool: NDArray[np.float64]
    total_torque: NDArray[np.float64]
    recovered_acceleration: NDArray[np.float64]


def _quintic_reference(
    start: NDArray[np.float64], end: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    duration = float(STUDY_TIME[-1] - STUDY_TIME[0])
    u = (STUDY_TIME - STUDY_TIME[0]) / duration
    blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    blend_rate = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / duration
    blend_acceleration = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / (
        duration**2
    )
    delta = end - start
    return (
        start + np.outer(blend, delta),
        np.outer(blend_rate, delta),
        np.outer(blend_acceleration, delta),
    )


# [dynamics-study-start]
def compute_dynamics_results() -> DynamicsResults:
    """Compute mass, torque, and round-trip histories through public APIs."""
    fixture = load_panda_fixture()
    theta, dtheta, ddtheta = _quintic_reference(START, MID)

    matrices = []
    inertia_rows = []
    velocity_rows = []
    gravity_rows = []
    tool_rows = []
    torque_rows = []
    recovered_rows = []
    for q, dq, ddq in zip(theta, dtheta, ddtheta):
        mass = assert_finite("mass matrix", fixture.dynamics.mass_matrix(q))
        velocity_force = assert_finite(
            "velocity force",
            fixture.dynamics.velocity_quadratic_forces(q, dq),
        )
        gravity = assert_finite(
            "gravity force", fixture.dynamics.gravity_forces(q, GRAVITY)
        )
        jacobian = assert_finite("space Jacobian", fixture.serial.jacobian(q))
        tool = assert_finite("tool torque", jacobian.T @ TOOL_WRENCH)
        inertia = assert_finite("inertial torque", mass @ ddq)
        total = assert_finite(
            "inverse dynamics torque",
            fixture.dynamics.inverse_dynamics(
                q, dq, ddq, GRAVITY, TOOL_WRENCH
            ),
        )
        recovered = assert_finite(
            "forward dynamics acceleration",
            fixture.dynamics.forward_dynamics(
                q, dq, total, GRAVITY, TOOL_WRENCH
            ),
        )
        matrices.append(mass)
        inertia_rows.append(inertia)
        velocity_rows.append(velocity_force)
        gravity_rows.append(gravity)
        tool_rows.append(tool)
        torque_rows.append(total)
        recovered_rows.append(recovered)

    return DynamicsResults(
        time=STUDY_TIME.copy(),
        theta=theta,
        velocity=dtheta,
        acceleration=ddtheta,
        mass_matrices=np.stack(matrices),
        inertia=np.stack(inertia_rows),
        velocity_force=np.stack(velocity_rows),
        gravity=np.stack(gravity_rows),
        tool=np.stack(tool_rows),
        total_torque=np.stack(torque_rows),
        recovered_acceleration=np.stack(recovered_rows),
    )


# [dynamics-study-end]


@dataclass(frozen=True)
class SingularityResults:
    """Jacobian spectra and linear velocity ellipsoids along one Panda path."""

    time: NDArray[np.float64]
    theta: NDArray[np.float64]
    singular_values: NDArray[np.float64]
    minimum_sigma: NDArray[np.float64]
    condition_number: NDArray[np.float64]
    linear_axes: NDArray[np.float64]
    ellipsoid_radii: NDArray[np.float64]
    near_singular: NDArray[np.bool_]
    public_status: NDArray[np.bool_]
    threshold: float


def compute_singularity_results() -> SingularityResults:
    """Compute public singularity diagnostics and a linear velocity ellipsoid."""
    fixture = load_panda_fixture()
    theta, _dtheta, _ddtheta = _quintic_reference(MID, NEAR_SINGULAR)
    analysis = Singularity(fixture.serial)
    spectra = []
    conditions = []
    axes = []
    radii = []
    statuses = []
    for q in theta:
        jacobian = assert_finite("space Jacobian", fixture.serial.jacobian(q))
        singular_values = assert_finite(
            "Jacobian singular values",
            np.linalg.svd(jacobian, compute_uv=False),
        )
        linear_u, linear_s, _linear_vh = np.linalg.svd(
            jacobian[3:, :], full_matrices=False
        )
        spectra.append(singular_values)
        axes.append(assert_finite("linear ellipsoid axes", linear_u))
        radii.append(assert_finite("linear ellipsoid radii", linear_s))
        conditions.append(float(analysis.condition_number(q)))
        statuses.append(bool(analysis.singularity_analysis(q)))

    singular_array = np.stack(spectra)
    minimum = singular_array[:, -1]
    near = minimum < SINGULARITY_THRESHOLD
    return SingularityResults(
        time=STUDY_TIME.copy(),
        theta=theta,
        singular_values=singular_array,
        minimum_sigma=minimum,
        condition_number=np.asarray(conditions, dtype=np.float64),
        linear_axes=np.stack(axes),
        ellipsoid_radii=np.stack(radii),
        near_singular=near,
        public_status=np.asarray(statuses, dtype=np.bool_),
        threshold=SINGULARITY_THRESHOLD,
    )
