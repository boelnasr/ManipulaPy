"""Deterministic Panda calculations for advanced documentation motion studies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ManipulaPy.ManipulaPy_data import get_robot_urdf
from ManipulaPy.control import ManipulatorController
from ManipulaPy.dynamics import ManipulatorDynamics
from ManipulaPy.kinematics import SerialManipulator
from ManipulaPy.path_planning import TrajectoryPlanning
from ManipulaPy.potential_field import PotentialField
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


@dataclass(frozen=True)
class TrajectorySeries:
    """Joint trajectory samples and their first three time derivatives."""

    positions: NDArray[np.float64]
    velocities: NDArray[np.float64]
    accelerations: NDArray[np.float64]
    jerk: NDArray[np.float64]


@dataclass(frozen=True)
class PlanningResults:
    """Interpolation and joint-space obstacle results for three studies."""

    time: NDArray[np.float64]
    cubic: TrajectorySeries
    quintic: TrajectorySeries
    joint_tool_path: NDArray[np.float64]
    cartesian_tool_path: NDArray[np.float64]
    nominal_path: NDArray[np.float64]
    corrected_path: NDArray[np.float64]
    obstacle_q: NDArray[np.float64]
    minimum_joint_clearance: float


class CollisionStudyUnavailable(RuntimeError):
    """Raised when the public potential-field path misses its study contract."""


def _trajectory_series(data: dict[str, Any]) -> TrajectorySeries:
    positions = assert_finite("joint positions", data["positions"])
    velocities = assert_finite("joint velocities", data["velocities"])
    accelerations = assert_finite("joint accelerations", data["accelerations"])
    dt = float(STUDY_TIME[1] - STUDY_TIME[0])
    jerk = assert_finite(
        "joint jerk", np.gradient(accelerations, dt, axis=0, edge_order=2)
    )
    return TrajectorySeries(positions, velocities, accelerations, jerk)


# [planning-study-start]
def compute_planning_results() -> PlanningResults:
    """Compute public joint, Cartesian, and potential-field planning results."""
    fixture = load_panda_fixture()
    planner = TrajectoryPlanning(
        fixture.serial,
        None,
        fixture.dynamics,
        fixture.joint_limits,
        use_cuda=False,
        auto_optimize=False,
    )
    cubic = _trajectory_series(
        planner.joint_trajectory(START, GOAL, 4.0, 61, method=3)
    )
    quintic = _trajectory_series(
        planner.joint_trajectory(START, GOAL, 4.0, 61, method=5)
    )

    start_pose = assert_finite(
        "start pose", fixture.serial.forward_kinematics(START)
    )
    goal_pose = assert_finite("goal pose", fixture.serial.forward_kinematics(GOAL))
    cartesian = planner.cartesian_trajectory(
        start_pose, goal_pose, 4.0, 61, method=5
    )
    cartesian_tool_path = assert_finite(
        "Cartesian tool path", cartesian["positions"]
    )
    joint_tool_path = np.stack(
        [
            assert_finite(
                "joint-space tool pose",
                fixture.serial.forward_kinematics(q),
            )[:3, 3]
            for q in quintic.positions
        ]
    )

    gripper = 0.02
    start_full = np.append(START, gripper)
    goal_full = np.append(GOAL, gripper)
    obstacle_full = np.append(OBSTACLE_Q, gripper)
    planner.potential_field = PotentialField(
        attractive_gain=1.0,
        repulsive_gain=0.001,
        influence_distance=0.5,
    )
    corrected_full = assert_finite(
        "joint-space corrected path",
        planner.plan_trajectory(start_full, goal_full, [obstacle_full]),
    )
    clearances = np.linalg.norm(corrected_full - obstacle_full, axis=1)
    minimum_clearance = float(np.min(clearances))
    if not np.allclose(corrected_full[-1], goal_full, atol=1e-8):
        raise CollisionStudyUnavailable("corrected path does not reach the goal")
    if minimum_clearance < JOINT_CLEARANCE:
        raise CollisionStudyUnavailable(
            f"joint-space clearance {minimum_clearance:.3f} rad is below "
            f"the required {JOINT_CLEARANCE:.3f} rad"
        )

    return PlanningResults(
        time=STUDY_TIME.copy(),
        cubic=cubic,
        quintic=quintic,
        joint_tool_path=joint_tool_path,
        cartesian_tool_path=cartesian_tool_path,
        nominal_path=np.linspace(START, GOAL, 6, dtype=np.float64),
        corrected_path=corrected_full[:, :7],
        obstacle_q=OBSTACLE_Q.copy(),
        minimum_joint_clearance=minimum_clearance,
    )


# [planning-study-end]


CONTROL_KP = np.array(
    [80.0, 80.0, 60.0, 60.0, 40.0, 40.0, 30.0], dtype=np.float64
)
CONTROL_KI = np.ones(7, dtype=np.float64)
CONTROL_KD = np.array(
    [18.0, 18.0, 14.0, 14.0, 10.0, 10.0, 8.0], dtype=np.float64
)
TORQUE_LIMITS = np.array(
    [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float64
)
CONTROL_SUBSTEPS = 2


@dataclass(frozen=True)
class ControlRun:
    """One controller response under the shared deterministic conditions."""

    time: NDArray[np.float64]
    reference: NDArray[np.float64]
    theta: NDArray[np.float64]
    velocity: NDArray[np.float64]
    torque: NDArray[np.float64]
    rms_error: float


@dataclass(frozen=True)
class ControlResults:
    """Three controller responses and public step-response metrics."""

    time: NDArray[np.float64]
    reference: NDArray[np.float64]
    runs: dict[str, ControlRun]
    target_joint: int
    metrics: dict[str, float]
    torque_limits: NDArray[np.float64]


def _disturbance(time_value: float) -> NDArray[np.float64]:
    disturbance = np.zeros(7, dtype=np.float64)
    disturbance[5] = 0.3 * np.sin(2.0 * np.pi * time_value / 4.0)
    return disturbance


def simulate_control(mode: str) -> ControlRun:
    """Simulate one public controller against the shared Panda dynamics plant."""
    if mode not in {"open_loop", "pid", "computed_torque"}:
        raise ValueError(f"unknown control study mode: {mode}")
    fixture = load_panda_fixture()
    controller = ManipulatorController(fixture.dynamics)
    reference = np.repeat(MID[np.newaxis, :], len(STUDY_TIME), axis=0)
    zero = np.zeros(7, dtype=np.float64)
    limits = np.asarray(fixture.joint_limits, dtype=np.float64)
    theta = START.copy()
    velocity = zero.copy()
    theta_rows = []
    velocity_rows = []
    torque_rows = []
    interval = float(STUDY_TIME[1] - STUDY_TIME[0])
    substep = interval / CONTROL_SUBSTEPS

    for index, time_value in enumerate(STUDY_TIME):
        theta_rows.append(theta.copy())
        velocity_rows.append(velocity.copy())
        command = zero.copy()
        if index < len(STUDY_TIME) - 1:
            for substep_index in range(CONTROL_SUBSTEPS):
                if mode == "open_loop":
                    command = np.asarray(
                        controller.feedforward_control(
                            MID, zero, zero, GRAVITY, TOOL_WRENCH
                        ),
                        dtype=np.float64,
                    )
                elif mode == "pid":
                    command = np.asarray(
                        controller.pid_control(
                            MID,
                            zero,
                            theta,
                            velocity,
                            substep,
                            CONTROL_KP,
                            CONTROL_KI,
                            CONTROL_KD,
                            i_clamp=0.5,
                        ),
                        dtype=np.float64,
                    )
                else:
                    command = np.asarray(
                        controller.computed_torque_control(
                            MID,
                            zero,
                            zero,
                            theta,
                            velocity,
                            GRAVITY,
                            substep,
                            CONTROL_KP,
                            CONTROL_KI,
                            CONTROL_KD,
                            i_clamp=0.5,
                        ),
                        dtype=np.float64,
                    )
                substep_time = time_value + substep_index * substep
                command = np.clip(
                    command + _disturbance(substep_time),
                    -TORQUE_LIMITS,
                    TORQUE_LIMITS,
                )
                acceleration = assert_finite(
                    f"{mode} plant acceleration",
                    fixture.dynamics.forward_dynamics(
                        theta,
                        velocity,
                        command,
                        GRAVITY,
                        TOOL_WRENCH,
                    ),
                )
                velocity = velocity + acceleration * substep
                proposed = theta + velocity * substep
                theta = np.clip(proposed, limits[:, 0], limits[:, 1])
                velocity = np.where(theta == proposed, velocity, 0.0)
        torque_rows.append(command.copy())

    theta_array = assert_finite(f"{mode} joint history", np.stack(theta_rows))
    velocity_array = assert_finite(
        f"{mode} velocity history", np.stack(velocity_rows)
    )
    torque_array = assert_finite(f"{mode} torque history", np.stack(torque_rows))
    return ControlRun(
        time=STUDY_TIME.copy(),
        reference=reference,
        theta=theta_array,
        velocity=velocity_array,
        torque=torque_array,
        rms_error=float(np.sqrt(np.mean((theta_array - reference) ** 2))),
    )


# [control-study-start]
def compute_control_results() -> ControlResults:
    """Compute equal-condition controller responses and public metrics."""
    runs = {
        mode: simulate_control(mode)
        for mode in ("open_loop", "pid", "computed_torque")
    }
    target_joint = 0
    response = runs["computed_torque"].theta[:, target_joint]
    set_point = float(MID[target_joint])
    metrics_controller = ManipulatorController(load_panda_fixture().dynamics)
    metrics = {
        "rise_time": metrics_controller.calculate_rise_time(
            STUDY_TIME, response, set_point
        ),
        "percent_overshoot": metrics_controller.calculate_percent_overshoot(
            response, set_point
        ),
        "settling_time": metrics_controller.calculate_settling_time(
            STUDY_TIME, response, set_point
        ),
        "steady_state_error": metrics_controller.calculate_steady_state_error(
            response, set_point
        ),
    }
    return ControlResults(
        time=STUDY_TIME.copy(),
        reference=runs["computed_torque"].reference.copy(),
        runs=runs,
        target_joint=target_joint,
        metrics=metrics,
        torque_limits=TORQUE_LIMITS.copy(),
    )


# [control-study-end]
