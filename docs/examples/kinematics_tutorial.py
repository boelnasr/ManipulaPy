"""Deterministic Panda kinematics calculations used by the tutorial gallery."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass(frozen=True)
class TutorialResults:
    """Calculated values shared by the written and animated tutorials."""

    joint_names: tuple[str, ...]
    pose: NDArray[np.float64]
    jacobian: NDArray[np.float64]
    twist: NDArray[np.float64]
    singular_values: NDArray[np.float64]
    ik_solution: NDArray[np.float64]
    ik_success: bool
    ik_iterations: int
    translation_residual: float
    rotation_residual: float


# [load-panda-start]
import numpy as np

from ManipulaPy.ManipulaPy_data import get_robot_urdf
from ManipulaPy.kinematics import SerialManipulator
from ManipulaPy.urdf.types import JointType
from ManipulaPy.urdf_processor import URDFToSerialManipulator

ARM_DOF = 7
HOME = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.785])
TARGET = np.array([0.15, -0.45, 0.1, -1.8, 0.05, 1.45, 0.65])
JOINT_RATES = np.array([0.05, -0.03, 0.02, 0.0, 0.01, -0.02, 0.03])


def load_panda() -> tuple[
    SerialManipulator,
    tuple[str, ...],
    tuple[tuple[float, float], ...],
    int,
]:
    """Load the Panda processor and return its seven arm joints and limits."""
    processor = URDFToSerialManipulator(
        get_robot_urdf("panda"), backend="builtin", load_meshes=False
    )
    arm_joints = tuple(
        joint
        for joint in processor.robot.actuated_joints
        if joint.joint_type is JointType.REVOLUTE
        and joint.name.startswith("panda_joint")
    )
    if len(arm_joints) != ARM_DOF:
        raise RuntimeError(f"Expected 7 Panda arm joints, found {len(arm_joints)}")
    names = tuple(joint.name for joint in arm_joints)
    if any(joint.limit is None for joint in arm_joints):
        raise RuntimeError("Every Panda arm joint must declare limits")
    limits = tuple((joint.limit.lower, joint.limit.upper) for joint in arm_joints)
    return processor.serial_manipulator, names, limits, processor.robot.num_dofs


# [load-panda-end]


def pose_residual(
    actual: NDArray[np.float64], desired: NDArray[np.float64]
) -> tuple[float, float]:
    """Return translation (metres) and rotation (radians) pose residuals."""
    translation = float(np.linalg.norm(actual[:3, 3] - desired[:3, 3]))
    relative_rotation = actual[:3, :3].T @ desired[:3, :3]
    cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
    return translation, float(np.arccos(cosine))


def _solve_to_target(robot, target_pose, max_iterations):
    """Run the public iterative solver with the tutorial's fixed settings."""
    return robot.iterative_inverse_kinematics(
        target_pose,
        HOME,
        max_iterations=max_iterations,
        adaptive_tuning=True,
        backtracking=True,
    )


# [forward-kinematics-start]
def forward_kinematics_step(robot, home, target):
    """Return the tool pose at ``home`` and the target pose for IK."""
    pose = np.asarray(robot.forward_kinematics(home, frame="space"), dtype=np.float64)
    target_pose = robot.forward_kinematics(target)
    return pose, target_pose


# [forward-kinematics-end]


# [velocity-kinematics-start]
def velocity_kinematics_step(robot, configuration, joint_rates):
    """Return the space Jacobian, tool twist, and Jacobian singular values."""
    jacobian = np.asarray(robot.jacobian(configuration, frame="space"), dtype=np.float64)
    twist = np.asarray(
        robot.end_effector_velocity(configuration, joint_rates, frame="space"),
        dtype=np.float64,
    )
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    return jacobian, twist, singular_values


# [velocity-kinematics-end]


# [inverse-kinematics-start]
def inverse_kinematics_step(robot, target_pose, initial_guess):
    """Solve for the Panda arm configuration that reaches ``target_pose``."""
    solution, success, iterations = robot.iterative_inverse_kinematics(
        target_pose,
        initial_guess,
        max_iterations=400,
        adaptive_tuning=True,
        backtracking=True,
    )
    ik_solution = np.asarray(solution, dtype=np.float64)
    return ik_solution, success, iterations


# [inverse-kinematics-end]


# [validation-start]
def validation_step(robot, solution, target_pose):
    """Return translation and rotation pose residuals for an IK solution."""
    solved_pose = robot.forward_kinematics(solution)
    return pose_residual(solved_pose, target_pose)


# [validation-end]


def compute_tutorial_results() -> TutorialResults:
    """Compute the deterministic values used in the kinematics tutorial."""
    robot, joint_names, _limits, _full_dof = load_panda()
    pose, target_pose = forward_kinematics_step(robot, HOME, TARGET)
    jacobian, twist, singular_values = velocity_kinematics_step(
        robot, HOME, JOINT_RATES
    )
    ik_solution, success, iterations = inverse_kinematics_step(
        robot, target_pose, HOME
    )
    translation_residual, rotation_residual = validation_step(
        robot, ik_solution, target_pose
    )

    return TutorialResults(
        joint_names=joint_names,
        pose=pose,
        jacobian=jacobian,
        twist=twist,
        singular_values=singular_values,
        ik_solution=ik_solution,
        ik_success=bool(success),
        ik_iterations=int(iterations),
        translation_residual=translation_residual,
        rotation_residual=rotation_residual,
    )


def compute_ik_trace(
    max_budget: int = 8,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """Return solver residuals for iteration budgets from one through ``max_budget``."""
    if max_budget < 1:
        raise ValueError("max_budget must be at least 1")

    robot, _joint_names, _limits, _full_dof = load_panda()
    target_pose = robot.forward_kinematics(TARGET)
    budgets = np.arange(1, max_budget + 1)
    translation_residuals = []
    rotation_residuals = []
    for budget in budgets:
        solution, _success, _iterations = _solve_to_target(
            robot, target_pose, int(budget)
        )
        solved_pose = robot.forward_kinematics(solution)
        translation, rotation = pose_residual(solved_pose, target_pose)
        translation_residuals.append(translation)
        rotation_residuals.append(rotation)

    return (
        budgets,
        np.asarray(translation_residuals, dtype=np.float64),
        np.asarray(rotation_residuals, dtype=np.float64),
    )
