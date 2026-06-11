"""Shared robot loading + conventions for the notebook course.

Running robot: Franka Emika Panda. The URDF has 7 revolute joints plus a fixed
``panda_joint8`` flange, so ManipulaPy's screw list has 8 columns while FK /
Jacobian accept either a 7- or 8-vector. The course standardises on the 7
actuated joints; see notebook 02 for the full explanation.
"""
from __future__ import annotations

import os

import numpy as np
from ManipulaPy import ManipulaPy_data
from ManipulaPy.urdf_processor import URDFToSerialManipulator

N_JOINTS = 7
PANDA_URDF = os.path.join(
    os.path.dirname(ManipulaPy_data.__file__), "franka_panda", "panda.urdf"
)

# A non-singular, visually clear default configuration (radians), 7 actuated joints.
HOME = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.785])


def load_panda():
    """Return ``(SerialManipulator, ManipulatorDynamics)`` for the Panda."""
    proc = URDFToSerialManipulator(PANDA_URDF)
    return proc.serial_manipulator, proc.dynamics


def joint_limits():
    """Conservative Panda joint limits (rad) for the 7 actuated joints."""
    return [
        (-2.90, 2.90),
        (-1.76, 1.76),
        (-2.90, 2.90),
        (-3.07, -0.07),
        (-2.90, 2.90),
        (-0.02, 3.75),
        (-2.90, 2.90),
    ]
