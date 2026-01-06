#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
URDF Data Types

Dataclass definitions for URDF elements, optimized for ManipulaPy.
Combines the best design patterns from yourdfpy, urchin, and urdfpy.

Copyright (c) 2025 Mohamed Aboelnasr
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union, Tuple
from enum import Enum
import numpy as np


def _array_eq(a: np.ndarray, b: np.ndarray, tol: float = 1e-10) -> bool:
    """Compare numpy arrays with tolerance."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if a.shape != b.shape:
        return False
    return np.allclose(a, b, atol=tol)


class JointType(Enum):
    """URDF joint types."""

    REVOLUTE = "revolute"
    CONTINUOUS = "continuous"
    PRISMATIC = "prismatic"
    FIXED = "fixed"
    FLOATING = "floating"
    PLANAR = "planar"

    @classmethod
    def from_string(cls, s: str) -> "JointType":
        """Create JointType from string."""
        try:
            return cls(s.lower())
        except ValueError:
            raise ValueError(f"Unknown joint type: {s}")


@dataclass
class Origin:
    """
    Transform origin (xyz position + rpy rotation).

    Represents the pose of a frame relative to another, combining
    translation (xyz) and rotation (roll-pitch-yaw Euler angles).
    """

    xyz: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    rpy: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    _matrix_cache: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.xyz = np.asarray(self.xyz, dtype=np.float64)
        self.rpy = np.asarray(self.rpy, dtype=np.float64)

    @property
    def matrix(self) -> np.ndarray:
        """
        Convert to 4x4 homogeneous transformation matrix.

        Uses ZYX (yaw-pitch-roll) Euler angle convention.
        """
        if self._matrix_cache is not None:
            return self._matrix_cache

        roll, pitch, yaw = self.rpy

        # Rotation matrices
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        # Combined rotation (ZYX order)
        R = np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=np.float64,
        )

        # Build 4x4 matrix
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = self.xyz

        self._matrix_cache = T
        return T

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> "Origin":
        """Create Origin from 4x4 transformation matrix."""
        xyz = T[:3, 3].copy()

        # Extract Euler angles (ZYX convention)
        R = T[:3, :3]
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))

        if np.abs(np.cos(pitch)) > 1e-10:
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            # Gimbal lock
            roll = np.arctan2(-R[1, 2], R[1, 1])
            yaw = 0.0

        return cls(xyz=xyz, rpy=np.array([roll, pitch, yaw]))

    @classmethod
    def identity(cls) -> "Origin":
        """Create identity transform."""
        return cls()

    def __eq__(self, other):
        if not isinstance(other, Origin):
            return NotImplemented
        return _array_eq(self.xyz, other.xyz) and _array_eq(self.rpy, other.rpy)

    def __hash__(self):
        return hash((tuple(self.xyz), tuple(self.rpy)))


@dataclass
class Inertial:
    """
    Link inertial properties.

    Contains mass, center of mass origin, and inertia tensor.
    """

    mass: float = 0.0
    origin: Origin = field(default_factory=Origin)
    inertia: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 3), dtype=np.float64)
    )

    def __post_init__(self):
        self.inertia = np.asarray(self.inertia, dtype=np.float64)

    @property
    def spatial_inertia(self) -> np.ndarray:
        """
        Return 6x6 spatial inertia matrix (G) for ManipulaPy.

        This computes the spatial inertia matrix at the link frame origin,
        accounting for the COM offset using the parallel axis theorem.

        Format: [[I_rotational, 0], [0, m*I_3x3]]

        If COM is at origin (no offset), returns simple block diagonal.
        If COM is offset, applies parallel axis theorem to transform
        the inertia tensor to the link frame.
        """
        G = np.zeros((6, 6), dtype=np.float64)

        # Get COM offset from origin
        com = self.origin.xyz

        if np.linalg.norm(com) < 1e-10:
            # No COM offset - simple case
            G[0:3, 0:3] = self.inertia
            G[3:6, 3:6] = self.mass * np.eye(3)
        else:
            # Apply parallel axis theorem to transform inertia to link frame
            # I_link = I_com + m * (|r|^2 * I - r * r^T)
            # where r is the vector from link origin to COM
            r = com
            r_sq = np.dot(r, r)
            r_outer = np.outer(r, r)

            # Parallel axis theorem
            I_at_origin = self.inertia + self.mass * (r_sq * np.eye(3) - r_outer)

            G[0:3, 0:3] = I_at_origin
            G[3:6, 3:6] = self.mass * np.eye(3)

        return G

    @property
    def spatial_inertia_at_com(self) -> np.ndarray:
        """
        Return 6x6 spatial inertia matrix at the center of mass.

        This is the inertia tensor as specified in the URDF, without
        parallel axis transformation.

        Format: [[I, 0], [0, m*I_3x3]]
        """
        G = np.zeros((6, 6), dtype=np.float64)
        G[0:3, 0:3] = self.inertia
        G[3:6, 3:6] = self.mass * np.eye(3)
        return G

    @classmethod
    def from_spatial_inertia(cls, G: np.ndarray, origin: Origin = None) -> "Inertial":
        """Create Inertial from 6x6 spatial inertia matrix."""
        inertia = G[0:3, 0:3].copy()
        mass = G[3, 3]
        return cls(mass=mass, origin=origin or Origin(), inertia=inertia)

    @property
    def com_position(self) -> np.ndarray:
        """Get center of mass position in link frame."""
        return self.origin.xyz.copy()

    def transform_to_frame(self, T: np.ndarray) -> "Inertial":
        """
        Transform inertial properties to a different frame.

        Args:
            T: 4x4 transformation matrix from current frame to new frame

        Returns:
            New Inertial object in the transformed frame
        """
        R = T[:3, :3]
        p = T[:3, 3]

        # Transform COM position
        new_com = R @ self.origin.xyz + p

        # Transform inertia tensor: I' = R * I * R^T
        new_inertia = R @ self.inertia @ R.T

        # Create new origin
        new_origin = Origin(xyz=new_com, rpy=self.origin.rpy)

        return Inertial(mass=self.mass, origin=new_origin, inertia=new_inertia)


# ==================== Geometry Types ====================


@dataclass
class Box:
    """Box geometry with size (x, y, z)."""

    size: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float64))

    def __post_init__(self):
        self.size = np.asarray(self.size, dtype=np.float64)

    def __eq__(self, other):
        if not isinstance(other, Box):
            return NotImplemented
        return _array_eq(self.size, other.size)

    def __hash__(self):
        return hash(tuple(self.size))


@dataclass
class Cylinder:
    """Cylinder geometry with radius and length."""

    radius: float = 1.0
    length: float = 1.0


@dataclass
class Sphere:
    """Sphere geometry with radius."""

    radius: float = 1.0


@dataclass
class Mesh:
    """
    Mesh geometry loaded from file.

    Supports STL, OBJ, DAE formats. Mesh data is lazy-loaded.
    """

    filename: str = ""
    scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float64))

    # Lazy-loaded mesh data
    _vertices: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _faces: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _trimesh: Optional[object] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.scale = np.asarray(self.scale, dtype=np.float64)
        if self.scale.size == 1:
            self.scale = np.full(3, self.scale.item(), dtype=np.float64)

    @property
    def vertices(self) -> Optional[np.ndarray]:
        """Mesh vertices (lazy-loaded)."""
        if self._vertices is None and self.filename:
            self._load_mesh()
        return self._vertices

    @property
    def faces(self) -> Optional[np.ndarray]:
        """Mesh faces (lazy-loaded)."""
        if self._faces is None and self.filename:
            self._load_mesh()
        return self._faces

    def _load_mesh(self) -> None:
        """Load mesh from file using trimesh."""
        try:
            import trimesh

            mesh = trimesh.load(self.filename, force="mesh")
            self._vertices = np.asarray(mesh.vertices, dtype=np.float64)
            self._faces = np.asarray(mesh.faces, dtype=np.int64)
            self._trimesh = mesh
        except ImportError:
            pass  # trimesh not available
        except Exception:
            pass  # Failed to load mesh

    def __eq__(self, other):
        if not isinstance(other, Mesh):
            return NotImplemented
        return self.filename == other.filename and _array_eq(self.scale, other.scale)

    def __hash__(self):
        return hash((self.filename, tuple(self.scale)))


# Union type for geometry
Geometry = Union[Box, Cylinder, Sphere, Mesh]


@dataclass
class Material:
    """Visual material with color and optional texture."""

    name: str = ""
    color: Optional[np.ndarray] = None  # RGBA
    texture: Optional[str] = None

    def __post_init__(self):
        if self.color is not None:
            self.color = np.asarray(self.color, dtype=np.float64)

    def __eq__(self, other):
        if not isinstance(other, Material):
            return NotImplemented
        return (
            self.name == other.name
            and _array_eq(self.color, other.color)
            and self.texture == other.texture
        )

    def __hash__(self):
        color_tuple = tuple(self.color) if self.color is not None else None
        return hash((self.name, color_tuple, self.texture))


@dataclass
class Visual:
    """Link visual properties."""

    name: str = ""
    origin: Origin = field(default_factory=Origin)
    geometry: Optional[Geometry] = None
    material: Optional[Material] = None


@dataclass
class Collision:
    """Link collision properties."""

    name: str = ""
    origin: Origin = field(default_factory=Origin)
    geometry: Optional[Geometry] = None


# ==================== Joint Properties ====================


@dataclass
class JointLimit:
    """Joint position, velocity, and effort limits."""

    lower: float = 0.0
    upper: float = 0.0
    effort: float = 0.0
    velocity: float = 0.0

    @property
    def range(self) -> Tuple[float, float]:
        """Return (lower, upper) tuple."""
        return (self.lower, self.upper)


@dataclass
class JointDynamics:
    """Joint dynamics parameters."""

    damping: float = 0.0
    friction: float = 0.0


@dataclass
class JointMimic:
    """Joint mimic constraints for coupled joints."""

    joint: str = ""
    multiplier: float = 1.0
    offset: float = 0.0


@dataclass
class SafetyController:
    """Joint safety controller parameters."""

    soft_lower_limit: float = 0.0
    soft_upper_limit: float = 0.0
    k_position: float = 0.0
    k_velocity: float = 0.0


@dataclass
class JointCalibration:
    """Joint calibration parameters."""

    rising: Optional[float] = None
    falling: Optional[float] = None


# ==================== Transmission Types ====================


@dataclass
class Actuator:
    """Actuator element within a transmission."""

    name: str
    mechanical_reduction: float = 1.0
    hardware_interface: Optional[str] = None


@dataclass
class TransmissionJoint:
    """Joint reference within a transmission."""

    name: str
    hardware_interface: Optional[str] = None


@dataclass
class Transmission:
    """
    URDF Transmission element.

    Describes the relationship between actuators and joints,
    including gear ratios and hardware interfaces.
    """

    name: str
    type: str = ""
    joints: List[TransmissionJoint] = field(default_factory=list)
    actuators: List[Actuator] = field(default_factory=list)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, Transmission):
            return NotImplemented
        return self.name == other.name


# ==================== Main URDF Elements ====================


@dataclass
class Link:
    """
    URDF Link element.

    Represents a rigid body with visual, collision, and inertial properties.
    """

    name: str
    inertial: Optional[Inertial] = None
    visuals: List[Visual] = field(default_factory=list)
    collisions: List[Collision] = field(default_factory=list)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, Link):
            return NotImplemented
        return self.name == other.name


@dataclass
class Joint:
    """
    URDF Joint element.

    Represents a connection between two links with specified kinematics.
    """

    name: str
    joint_type: JointType
    parent: str  # Parent link name
    child: str  # Child link name
    origin: Origin = field(default_factory=Origin)
    axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    limit: Optional[JointLimit] = None
    dynamics: Optional[JointDynamics] = None
    mimic: Optional[JointMimic] = None
    safety: Optional[SafetyController] = None
    calibration: Optional[JointCalibration] = None

    def __post_init__(self):
        self.axis = np.asarray(self.axis, dtype=np.float64)
        # Normalize axis
        norm = np.linalg.norm(self.axis)
        if norm > 1e-10:
            self.axis = self.axis / norm

    @property
    def is_actuated(self) -> bool:
        """Check if joint is actuated (not fixed)."""
        return self.joint_type != JointType.FIXED

    @property
    def is_mimic(self) -> bool:
        """Check if joint mimics another."""
        return self.mimic is not None

    def get_child_pose(self, q=None) -> np.ndarray:
        """
        Compute child link pose relative to parent for given configuration.

        Args:
            q: Joint configuration:
               - float for revolute/continuous (angle in rad)
               - float for prismatic (displacement in m)
               - array [x, y, theta] for planar (2D position + rotation)
               - array [x, y, z, qx, qy, qz, qw] for floating (3D pose as quaternion)
               - None defaults to zero configuration

        Returns:
            4x4 homogeneous transformation matrix
        """
        T_origin = self.origin.matrix

        if self.joint_type == JointType.FIXED:
            return T_origin

        if self.joint_type in (JointType.REVOLUTE, JointType.CONTINUOUS):
            angle = float(q) if q is not None else 0.0
            R = self._axis_angle_rotation(self.axis, angle)
            T_joint = np.eye(4, dtype=np.float64)
            T_joint[:3, :3] = R
            return T_origin @ T_joint

        if self.joint_type == JointType.PRISMATIC:
            disp = float(q) if q is not None else 0.0
            T_joint = np.eye(4, dtype=np.float64)
            T_joint[:3, 3] = self.axis * disp
            return T_origin @ T_joint

        if self.joint_type == JointType.PLANAR:
            # Planar joint: translation in XY plane + rotation around Z
            # q = [x, y, theta] or scalar (theta only)
            T_joint = np.eye(4, dtype=np.float64)
            if q is not None:
                q = np.atleast_1d(q)
                if len(q) >= 2:
                    T_joint[0, 3] = q[0]  # x translation
                    T_joint[1, 3] = q[1]  # y translation
                if len(q) >= 3:
                    # Rotation around z-axis
                    theta = q[2]
                    c, s = np.cos(theta), np.sin(theta)
                    T_joint[0, 0] = c
                    T_joint[0, 1] = -s
                    T_joint[1, 0] = s
                    T_joint[1, 1] = c
            return T_origin @ T_joint

        if self.joint_type == JointType.FLOATING:
            # Floating joint: full 6-DOF pose
            # q = [x, y, z, qx, qy, qz, qw] (position + quaternion)
            T_joint = np.eye(4, dtype=np.float64)
            if q is not None:
                q = np.atleast_1d(q)
                if len(q) >= 3:
                    T_joint[:3, 3] = q[:3]  # position
                if len(q) >= 7:
                    # Quaternion to rotation matrix
                    T_joint[:3, :3] = self._quaternion_to_rotation(q[3:7])
            return T_origin @ T_joint

        # Unknown joint type - return origin only
        return T_origin

    @staticmethod
    def _quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix.

        Args:
            q: Quaternion as [qx, qy, qz, qw]

        Returns:
            3x3 rotation matrix
        """
        qx, qy, qz, qw = q
        # Normalize
        n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if n > 1e-10:
            qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

        return np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ], dtype=np.float64)

    def get_child_poses_batch(self, q: np.ndarray) -> np.ndarray:
        """
        Batch compute child poses for multiple configurations.

        Optimized vectorized implementation.

        Args:
            q: Array of joint configurations (N,)

        Returns:
            Array of 4x4 transformation matrices (N, 4, 4)
        """
        n = len(q)
        T_origin = self.origin.matrix

        if self.joint_type == JointType.FIXED:
            return np.tile(T_origin, (n, 1, 1))

        if self.joint_type in (JointType.REVOLUTE, JointType.CONTINUOUS):
            R_batch = self._axis_angle_rotation_batch(self.axis, q)
            T_joint = np.zeros((n, 4, 4), dtype=np.float64)
            T_joint[:, :3, :3] = R_batch
            T_joint[:, 3, 3] = 1.0
            return np.matmul(T_origin, T_joint)

        if self.joint_type == JointType.PRISMATIC:
            T_joint = np.tile(np.eye(4), (n, 1, 1))
            T_joint[:, :3, 3] = np.outer(q, self.axis)
            return np.matmul(T_origin, T_joint)

        return np.tile(T_origin, (n, 1, 1))

    @staticmethod
    def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
        """Rodrigues' rotation formula."""
        axis = axis / np.linalg.norm(axis)
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
            dtype=np.float64,
        )
        c, s = np.cos(angle), np.sin(angle)
        return np.eye(3) + s * K + (1 - c) * (K @ K)

    @staticmethod
    def _axis_angle_rotation_batch(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """
        Batch Rodrigues' rotation formula.

        Optimized for computing many rotations about same axis.
        """
        axis = axis / np.linalg.norm(axis)
        n = len(angles)

        sina = np.sin(angles)
        cosa = np.cos(angles)

        # Skew-symmetric matrix for axis
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
            dtype=np.float64,
        )
        K2 = K @ K
        outer = np.outer(axis, axis)

        # R = I + sin(a)*K + (1-cos(a))*K^2
        # R = cos(a)*I + sin(a)*K + (1-cos(a))*axis*axis^T
        R = np.zeros((n, 3, 3), dtype=np.float64)
        R[:] = np.eye(3)
        R *= cosa[:, np.newaxis, np.newaxis]
        R += sina[:, np.newaxis, np.newaxis] * K
        R += (1 - cosa)[:, np.newaxis, np.newaxis] * outer

        return R

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, Joint):
            return NotImplemented
        return self.name == other.name
