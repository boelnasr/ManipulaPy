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
