#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
URDF Core Class

Main URDF class providing forward kinematics, ManipulaPy integration,
and visualization capabilities.

Copyright (c) 2025 Mohamed Aboelnasr
"""

from typing import Optional, Dict, List, Union, Callable, Tuple
from pathlib import Path
from collections import OrderedDict
import numpy as np

from .types import Link, Joint, JointType, Material, Transmission


class URDF:
    """
    ManipulaPy native URDF parser.

    Optimized for robotics kinematics and dynamics workflows with
    direct integration to SerialManipulator and ManipulatorDynamics.

    Example:
        >>> robot = URDF.load("robot.urdf")
        >>> robot.num_actuated_joints
        6
        >>> fk = robot.link_fk()
        >>> manipulator = robot.to_serial_manipulator()
    """

    def __init__(
        self,
        name: str = "robot",
        links: Optional[List[Link]] = None,
        joints: Optional[List[Joint]] = None,
        materials: Optional[Dict[str, Material]] = None,
        transmissions: Optional[List[Transmission]] = None,
        filename_handler: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize URDF.

        Args:
            name: Robot name
            links: List of Link objects
            joints: List of Joint objects
            materials: Material definitions
            transmissions: List of Transmission objects
            filename_handler: Function to resolve mesh filenames
        """
        self.name = name
        self._filename_handler = filename_handler or (lambda x: x)

        # Build internal maps
        self._links: Dict[str, Link] = {}
        self._joints: Dict[str, Joint] = {}
        self._materials: Dict[str, Material] = materials or {}
        self._transmissions: Dict[str, Transmission] = {}

        if links:
            for link in links:
                if link.name in self._links:
                    raise ValueError(f"Duplicate link name: {link.name}")
                self._links[link.name] = link

        if joints:
            for joint in joints:
                if joint.name in self._joints:
                    raise ValueError(f"Duplicate joint name: {joint.name}")
                self._joints[joint.name] = joint

        if transmissions:
            for transmission in transmissions:
                if transmission.name in self._transmissions:
                    raise ValueError(f"Duplicate transmission name: {transmission.name}")
                self._transmissions[transmission.name] = transmission

        # Cached computations
        self._kinematic_chain: Optional[List[Joint]] = None
        self._root_link_name: Optional[str] = None  # primary root (for backward compatibility)
        self._root_link_names: Optional[List[str]] = None  # all roots
        self._end_link_names: Optional[List[str]] = None
        self._cfg: Dict[str, float] = {}
        self._fk_cache: Dict[str, np.ndarray] = {}

        # Build kinematic structure
        self._build_kinematic_structure()

    # ==================== Loading ====================

    @classmethod
    def load(
        cls,
        filename: Union[str, Path],
        load_meshes: bool = False,
        mesh_dir: Optional[Union[str, Path]] = None,
        backend: str = "builtin",
    ) -> "URDF":
        """
        Load URDF from file.

        Args:
            filename: Path to URDF or XACRO file
            load_meshes: Load mesh geometry data
            mesh_dir: Base directory for mesh file resolution
            backend: Parser backend - "builtin" (default), "urchin", or "pybullet"

        Returns:
            URDF object

        Note:
            - "builtin": Native ManipulaPy parser (NumPy 2.0 compatible)
            - "urchin": Legacy urchin parser (requires urchin package)
            - "pybullet": PyBullet-based parser (requires pybullet package)
        """
        if backend == "builtin":
            return cls._load_builtin(filename, load_meshes=load_meshes, mesh_dir=mesh_dir)
        elif backend == "urchin":
            return cls._load_urchin(filename)
        elif backend == "pybullet":
            return cls._load_pybullet(filename)
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'builtin', 'urchin', or 'pybullet'")

    @classmethod
    def _load_builtin(
        cls,
        filename: Union[str, Path],
        load_meshes: bool = False,
        mesh_dir: Optional[Union[str, Path]] = None,
    ) -> "URDF":
        """Load URDF using native builtin parser."""
        from .parser import URDFParser

        mesh_path = Path(mesh_dir) if mesh_dir else None
        return URDFParser.parse_file(filename, load_meshes=load_meshes, mesh_dir=mesh_path)

    @classmethod
    def _load_urchin(cls, filename: Union[str, Path]) -> "URDF":
        """
        Load URDF using urchin parser (legacy fallback).

        Requires: pip install urchin
        Note: urchin is not compatible with NumPy 2.0+
        """
        try:
            import urchin
        except ImportError:
            raise ImportError(
                "urchin package not installed. Install with: pip install urchin\n"
                "Note: urchin is not compatible with NumPy 2.0+. "
                "Consider using backend='builtin' instead."
            )

        import warnings
        warnings.warn(
            "Using urchin backend which is not compatible with NumPy 2.0+. "
            "Consider using backend='builtin' for better compatibility.",
            DeprecationWarning,
            stacklevel=3,
        )

        # Load with urchin
        urchin_robot = urchin.URDF.load(str(filename))

        # Convert to our URDF format
        return cls._convert_from_urchin(urchin_robot)

    @classmethod
    def _convert_from_urchin(cls, urchin_robot) -> "URDF":
        """Convert urchin URDF to native URDF."""
        from .types import (
            Link, Joint, JointType, Origin, Inertial, Visual, Collision,
            JointLimit, JointDynamics, JointMimic, Material, Box, Cylinder, Sphere, Mesh,
        )

        # Convert links
        links = []
        for ulink in urchin_robot.links:
            # Convert inertial
            inertial = None
            if ulink.inertial is not None:
                origin = Origin(
                    xyz=np.array(ulink.inertial.origin.xyz) if ulink.inertial.origin else np.zeros(3),
                    rpy=np.array(ulink.inertial.origin.rpy) if ulink.inertial.origin else np.zeros(3),
                )
                inertial = Inertial(
                    mass=ulink.inertial.mass,
                    inertia=ulink.inertial.inertia if hasattr(ulink.inertial, 'inertia') else np.eye(3),
                    origin=origin,
                )

            # Convert visuals
            visuals = []
            for uvis in (ulink.visuals or []):
                vis_origin = Origin(
                    xyz=np.array(uvis.origin.xyz) if uvis.origin else np.zeros(3),
                    rpy=np.array(uvis.origin.rpy) if uvis.origin else np.zeros(3),
                )
                geometry = cls._convert_urchin_geometry(uvis.geometry) if uvis.geometry else None
                visuals.append(Visual(origin=vis_origin, geometry=geometry))

            # Convert collisions
            collisions = []
            for ucol in (ulink.collisions or []):
                col_origin = Origin(
                    xyz=np.array(ucol.origin.xyz) if ucol.origin else np.zeros(3),
                    rpy=np.array(ucol.origin.rpy) if ucol.origin else np.zeros(3),
                )
                geometry = cls._convert_urchin_geometry(ucol.geometry) if ucol.geometry else None
                collisions.append(Collision(origin=col_origin, geometry=geometry))

            links.append(Link(
                name=ulink.name,
                inertial=inertial,
                visuals=visuals,
                collisions=collisions,
            ))

        # Convert joints
        joints = []
        for ujoint in urchin_robot.joints:
            # Map joint type
            jtype_map = {
                'revolute': JointType.REVOLUTE,
                'continuous': JointType.CONTINUOUS,
                'prismatic': JointType.PRISMATIC,
                'fixed': JointType.FIXED,
                'floating': JointType.FLOATING,
                'planar': JointType.PLANAR,
            }
            joint_type = jtype_map.get(ujoint.joint_type, JointType.FIXED)

            origin = Origin(
                xyz=np.array(ujoint.origin.xyz) if ujoint.origin else np.zeros(3),
                rpy=np.array(ujoint.origin.rpy) if ujoint.origin else np.zeros(3),
            )

            axis = np.array(ujoint.axis) if ujoint.axis is not None else np.array([0, 0, 1])

            # Convert limit
            limit = None
            if ujoint.limit is not None:
                limit = JointLimit(
                    lower=ujoint.limit.lower if hasattr(ujoint.limit, 'lower') else 0.0,
                    upper=ujoint.limit.upper if hasattr(ujoint.limit, 'upper') else 0.0,
                    velocity=getattr(ujoint.limit, 'velocity', None),
                    effort=getattr(ujoint.limit, 'effort', None),
                )

            # Convert dynamics
            dynamics = None
            if ujoint.dynamics is not None:
                dynamics = JointDynamics(
                    damping=getattr(ujoint.dynamics, 'damping', 0.0),
                    friction=getattr(ujoint.dynamics, 'friction', 0.0),
                )

            # Convert mimic
            mimic = None
            if ujoint.mimic is not None:
                mimic = JointMimic(
                    joint=ujoint.mimic.joint,
                    multiplier=getattr(ujoint.mimic, 'multiplier', 1.0),
                    offset=getattr(ujoint.mimic, 'offset', 0.0),
                )

            joints.append(Joint(
                name=ujoint.name,
                joint_type=joint_type,
                parent=ujoint.parent,
                child=ujoint.child,
                origin=origin,
                axis=axis,
                limit=limit,
                dynamics=dynamics,
                mimic=mimic,
            ))

        return cls(
            name=urchin_robot.name or "robot",
            links=links,
            joints=joints,
        )

    @classmethod
    def _convert_urchin_geometry(cls, ugeom):
        """Convert urchin geometry to native geometry."""
        from .types import Box, Cylinder, Sphere, Mesh

        geom_type = type(ugeom).__name__.lower()

        if geom_type == 'box':
            return Box(size=np.array(ugeom.size))
        elif geom_type == 'cylinder':
            return Cylinder(radius=ugeom.radius, length=ugeom.length)
        elif geom_type == 'sphere':
            return Sphere(radius=ugeom.radius)
        elif geom_type == 'mesh':
            scale = np.array(ugeom.scale) if ugeom.scale is not None else np.ones(3)
            return Mesh(filename=ugeom.filename, scale=scale)

        return None

    @classmethod
    def _load_pybullet(cls, filename: Union[str, Path]) -> "URDF":
        """
        Load URDF using PyBullet parser.

        Requires: pip install pybullet
        """
        try:
            import pybullet as p
            import pybullet_data
        except ImportError:
            raise ImportError(
                "pybullet package not installed. Install with: pip install pybullet"
            )

        from .types import (
            Link, Joint, JointType, Origin, Inertial, JointLimit,
        )

        # Connect to PyBullet in DIRECT mode (no GUI)
        physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        try:
            robot_id = p.loadURDF(str(filename))

            # Get robot info
            num_joints = p.getNumJoints(robot_id)

            # Build links and joints
            links = []
            joints = []

            # Add base link
            base_name = p.getBodyInfo(robot_id)[0].decode('utf-8')
            # Get base dynamics info
            base_dynamics = p.getDynamicsInfo(robot_id, -1)
            base_mass = base_dynamics[0]
            base_inertia_diag = base_dynamics[2]
            base_inertial = Inertial(
                mass=base_mass,
                inertia=np.diag(base_inertia_diag),
            ) if base_mass > 0 else None

            links.append(Link(name=base_name, inertial=base_inertial))

            # Map PyBullet joint types
            pb_joint_type_map = {
                p.JOINT_REVOLUTE: JointType.REVOLUTE,
                p.JOINT_PRISMATIC: JointType.PRISMATIC,
                p.JOINT_SPHERICAL: JointType.FLOATING,
                p.JOINT_PLANAR: JointType.PLANAR,
                p.JOINT_FIXED: JointType.FIXED,
            }

            for i in range(num_joints):
                joint_info = p.getJointInfo(robot_id, i)

                joint_name = joint_info[1].decode('utf-8')
                joint_type_pb = joint_info[2]
                link_name = joint_info[12].decode('utf-8')
                parent_idx = joint_info[16]
                parent_name = base_name if parent_idx == -1 else p.getJointInfo(robot_id, parent_idx)[12].decode('utf-8')

                # Joint position/orientation in parent frame
                joint_pos = np.array(joint_info[14])
                joint_orn = np.array(joint_info[15])
                # Convert quaternion to rpy
                rpy = np.array(p.getEulerFromQuaternion(joint_orn))

                # Joint axis
                axis = np.array(joint_info[13])

                # Joint limits
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                max_velocity = joint_info[11]
                max_force = joint_info[10]

                joint_type = pb_joint_type_map.get(joint_type_pb, JointType.FIXED)

                # Get link dynamics info
                dynamics = p.getDynamicsInfo(robot_id, i)
                mass = dynamics[0]
                inertia_diag = dynamics[2]
                local_inertial_pos = np.array(dynamics[3])
                local_inertial_orn = dynamics[4]
                local_inertial_rpy = np.array(p.getEulerFromQuaternion(local_inertial_orn))

                inertial = Inertial(
                    mass=mass,
                    inertia=np.diag(inertia_diag),
                    origin=Origin(xyz=local_inertial_pos, rpy=local_inertial_rpy),
                ) if mass > 0 else None

                links.append(Link(name=link_name, inertial=inertial))

                limit = JointLimit(
                    lower=lower_limit,
                    upper=upper_limit,
                    velocity=max_velocity if max_velocity > 0 else None,
                    effort=max_force if max_force > 0 else None,
                ) if joint_type != JointType.FIXED else None

                joints.append(Joint(
                    name=joint_name,
                    joint_type=joint_type,
                    parent=parent_name,
                    child=link_name,
                    origin=Origin(xyz=joint_pos, rpy=rpy),
                    axis=axis,
                    limit=limit,
                ))

        finally:
            p.disconnect(physics_client)

        return cls(
            name=Path(filename).stem,
            links=links,
            joints=joints,
        )

    @classmethod
    def from_xml_string(cls, xml_string: str, **kwargs) -> "URDF":
        """Load URDF from XML string."""
        from .parser import URDFParser

        return URDFParser.parse_string(xml_string, **kwargs)

    # ==================== Properties ====================

    @property
    def links(self) -> List[Link]:
        """All links in order."""
        return list(self._links.values())

    @property
    def joints(self) -> List[Joint]:
        """All joints in order."""
        return list(self._joints.values())

    @property
    def transmissions(self) -> List[Transmission]:
        """All transmissions."""
        return list(self._transmissions.values())

    @property
    def transmission_map(self) -> Dict[str, Transmission]:
        """Transmission name to Transmission mapping."""
        return self._transmissions

    @property
    def link_map(self) -> Dict[str, Link]:
        """Link name to Link mapping."""
        return self._links

    @property
    def joint_map(self) -> Dict[str, Joint]:
        """Joint name to Joint mapping."""
        return self._joints

    @property
    def actuated_joints(self) -> List[Joint]:
        """Non-fixed, non-mimic joints in kinematic chain order."""
        if self._kinematic_chain is None:
            self._build_kinematic_structure()
        return [j for j in self._kinematic_chain if j.is_actuated and not j.is_mimic]

    @property
    def actuated_joint_names(self) -> List[str]:
        """Names of actuated joints."""
        return [j.name for j in self.actuated_joints]

    @property
    def num_actuated_joints(self) -> int:
        """Number of actuated (non-fixed, non-mimic) joints."""
        return len(self.actuated_joints)

    @property
    def num_dofs(self) -> int:
        """Alias for num_actuated_joints."""
        return self.num_actuated_joints

    @property
    def root_link(self) -> Link:
        """Root link of the kinematic tree."""
        if self._root_link_name is None:
            self._build_kinematic_structure()
        return self._links[self._root_link_name]

    @property
    def root_links(self) -> List[Link]:
        """All root links (supports multi-root URDFs)."""
        if self._root_link_names is None:
            self._build_kinematic_structure()
        return [self._links[name] for name in self._root_link_names]

    @property
    def end_links(self) -> List[Link]:
        """End effector (leaf) links."""
        if self._end_link_names is None:
            self._build_kinematic_structure()
        return [self._links[name] for name in self._end_link_names]

    @property
    def end_effector_link(self) -> Link:
        """Primary end effector link (first end link)."""
        end_links = self.end_links
        return end_links[0] if end_links else self.root_link

    @property
    def kinematic_chain(self) -> List[Joint]:
        """Ordered list of joints from root to leaves."""
        if self._kinematic_chain is None:
            self._build_kinematic_structure()
        return self._kinematic_chain

    @property
    def joint_limits(self) -> List[Tuple[float, float]]:
        """Joint limits as list of (lower, upper) tuples."""
        limits = []
        for joint in self.actuated_joints:
            if joint.limit:
                limits.append((joint.limit.lower, joint.limit.upper))
            else:
                # Default for continuous joints
                limits.append((-np.pi, np.pi))
        return limits

    @property
    def cfg(self) -> np.ndarray:
        """Current configuration as array."""
        return np.array(
            [self._cfg.get(j.name, 0.0) for j in self.actuated_joints],
            dtype=np.float64,
        )

    @cfg.setter
    def cfg(self, value: Union[np.ndarray, List[float], Dict[str, float]]):
        """Set current configuration."""
        self.update_cfg(value)

    # ==================== Kinematic Structure ====================

    def _build_kinematic_structure(self) -> None:
        """Build kinematic tree structure."""
        if not self._joints:
            # Single link robot
            if self._links:
                self._root_link_name = list(self._links.keys())[0]
                self._root_link_names = [self._root_link_name]
                self._end_link_names = [self._root_link_name]
            self._kinematic_chain = []
            return

        # Find all child links
        child_links = {j.child for j in self._joints.values()}
        parent_links = {j.parent for j in self._joints.values()}

        # Root = parent that is never a child
        root_candidates = parent_links - child_links
        if not root_candidates:
            raise ValueError("URDF has no root link (cyclic structure?)")

        # Allow multiple roots: keep all, choose first as primary for legacy APIs
        self._root_link_names = list(root_candidates)
        self._root_link_name = self._root_link_names[0]
        if len(self._root_link_names) > 1:
            import warnings
            warnings.warn(
                f"URDF has multiple roots: {self._root_link_names}. "
                "Using the first as primary; all roots will be included in FK.",
                UserWarning,
            )

        # End links = children that are never parents
        all_links = set(self._links.keys())
        self._end_link_names = list(all_links - parent_links)
        if not self._end_link_names:
            # All links are parents - find leaves in joint tree
            self._end_link_names = list(child_links - parent_links)

        # Build kinematic chain via BFS from root
        chain = []
        visited = set(self._root_link_names)
        queue = list(self._root_link_names)

        # Build parent->joint mapping
        parent_to_joints: Dict[str, List[Joint]] = {}
        for joint in self._joints.values():
            if joint.parent not in parent_to_joints:
                parent_to_joints[joint.parent] = []
            parent_to_joints[joint.parent].append(joint)

        while queue:
            current = queue.pop(0)

            if current in parent_to_joints:
                for joint in parent_to_joints[current]:
                    if joint.child not in visited:
                        chain.append(joint)
                        visited.add(joint.child)
                        queue.append(joint.child)

        self._kinematic_chain = chain

        # Initialize configuration
        for joint in self.actuated_joints:
            if joint.name not in self._cfg:
                self._cfg[joint.name] = 0.0

    # ==================== Configuration ====================

    def update_cfg(
        self, cfg: Union[np.ndarray, List[float], Dict[str, float], None]
    ) -> None:
        """
        Update joint configuration.

        Args:
            cfg: Configuration as dict {joint_name: value}, array, or list
        """
        self._fk_cache.clear()

        if cfg is None:
            return

        if isinstance(cfg, dict):
            self._cfg.update(cfg)
        else:
            cfg_array = np.asarray(cfg, dtype=np.float64).flatten()
            if len(cfg_array) != self.num_actuated_joints:
                raise ValueError(
                    f"Configuration length {len(cfg_array)} != "
                    f"num_actuated_joints {self.num_actuated_joints}"
                )
            for i, joint in enumerate(self.actuated_joints):
                self._cfg[joint.name] = cfg_array[i]

    def _get_joint_cfg(self, joint: Joint) -> float:
        """Get configuration value for a joint, handling mimic joints."""
        if joint.mimic:
            mimic_cfg = self._cfg.get(joint.mimic.joint, 0.0)
            return mimic_cfg * joint.mimic.multiplier + joint.mimic.offset
        return self._cfg.get(joint.name, 0.0)

    # ==================== Forward Kinematics ====================

    def link_fk(
        self,
        cfg: Union[np.ndarray, List[float], Dict[str, float], None] = None,
        links: Optional[List[str]] = None,
        use_names: bool = False,
    ) -> Union[Dict[Link, np.ndarray], Dict[str, np.ndarray]]:
        """
        Compute forward kinematics for links.

        Args:
            cfg: Joint configuration (uses current if None)
            links: Specific link names to compute (None = all)
            use_names: Return dict with string keys instead of Link keys

        Returns:
            Dict mapping links (or link names) to 4x4 transformation matrices
        """
        if cfg is not None:
            self.update_cfg(cfg)

        # Compute transforms for all links (support multi-root)
        transforms: Dict[str, np.ndarray] = {
            root: np.eye(4, dtype=np.float64) for root in self._root_link_names
        }

        for joint in self._kinematic_chain:
            parent_T = transforms.get(joint.parent, np.eye(4, dtype=np.float64))
            q = self._get_joint_cfg(joint)
            joint_T = joint.get_child_pose(q)
            transforms[joint.child] = parent_T @ joint_T

        # Filter if specific links requested
        if links is not None:
            transforms = {k: v for k, v in transforms.items() if k in links}

        if use_names:
            return transforms
        else:
            return OrderedDict(
                (self._links[name], T)
                for name, T in transforms.items()
                if name in self._links
            )

    def link_fk_batch(
        self,
        cfgs: np.ndarray,
        links: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Batch compute forward kinematics for multiple configurations.

        Optimized vectorized implementation.

        Args:
            cfgs: Array of configurations (N, num_dofs)
            links: Specific link names to compute (None = all)

        Returns:
            Dict mapping link names to (N, 4, 4) transformation arrays
        """
        cfgs = np.asarray(cfgs, dtype=np.float64)
        if cfgs.ndim == 1:
            cfgs = cfgs.reshape(1, -1)

        n_cfgs = cfgs.shape[0]

        if cfgs.shape[1] != self.num_actuated_joints:
            raise ValueError(
                f"Configuration columns {cfgs.shape[1]} != "
                f"num_actuated_joints {self.num_actuated_joints}"
            )

        # Build joint name to column index mapping
        joint_idx = {j.name: i for i, j in enumerate(self.actuated_joints)}

        # Initialize transforms for each root (multi-root support)
        transforms: Dict[str, np.ndarray] = {
            root: np.tile(np.eye(4), (n_cfgs, 1, 1)) for root in self._root_link_names
        }

        for joint in self._kinematic_chain:
            parent_T = transforms.get(
                joint.parent, np.tile(np.eye(4), (n_cfgs, 1, 1))
            )

            # Get configuration for this joint
            if joint.mimic:
                mimic_idx = joint_idx.get(joint.mimic.joint, 0)
                q = cfgs[:, mimic_idx] * joint.mimic.multiplier + joint.mimic.offset
            elif joint.name in joint_idx:
                q = cfgs[:, joint_idx[joint.name]]
            else:
                q = np.zeros(n_cfgs, dtype=np.float64)

            joint_T = joint.get_child_poses_batch(q)
            transforms[joint.child] = np.matmul(parent_T, joint_T)

        # Filter if specific links requested
        if links is not None:
            transforms = {k: v for k, v in transforms.items() if k in links}

        return transforms

    def get_transform(
        self,
        frame_to: str,
        frame_from: str = "world",
        cfg: Optional[Union[np.ndarray, Dict[str, float]]] = None,
    ) -> np.ndarray:
        """
        Get transform between two frames.

        Args:
            frame_to: Target frame (link name)
            frame_from: Source frame (link name or "world")
            cfg: Joint configuration

        Returns:
            4x4 transformation matrix from frame_from to frame_to
        """
        fk = self.link_fk(cfg, use_names=True)

        if frame_to not in fk:
            raise ValueError(f"Unknown frame: {frame_to}")

        T_to = fk[frame_to]

        if frame_from == "world":
            return T_to

        if frame_from not in fk:
            raise ValueError(f"Unknown frame: {frame_from}")

        T_from = fk[frame_from]
        return np.linalg.inv(T_from) @ T_to

    # ==================== ManipulaPy Integration ====================

    def extract_screw_axes(self, tip_link: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Extract screw axis parameters for ManipulaPy.

        Args:
            tip_link: Optional link name to use as end-effector. Defaults to first end link.

        Returns:
            Dict with keys: M, S_list, B_list, G_list, omega_list, r_list, joint_limits
        """
        from ..utils import adjoint_transform

        actuated = self.actuated_joints
        n = len(actuated)

        if n == 0:
            raise ValueError("No actuated joints found")

        # Compute FK at home position (all zeros)
        home_cfg = {j.name: 0.0 for j in actuated}
        fk = self.link_fk(home_cfg, use_names=True)

        # End-effector home position (selectable)
        ee_link = tip_link if tip_link is not None else self.end_effector_link.name
        if ee_link not in fk:
            raise ValueError(f"tip_link '{ee_link}' not found among links")
        M = fk[ee_link].copy()

        # Extract screw axes in space frame
        S_list = np.zeros((6, n), dtype=np.float64)
        omega_list = np.zeros((3, n), dtype=np.float64)
        r_list = np.zeros((3, n), dtype=np.float64)
        G_list = []

        for i, joint in enumerate(actuated):
            if joint.joint_type in (JointType.PLANAR, JointType.FLOATING):
                raise ValueError(
                    f"Joint '{joint.name}' is {joint.joint_type.name.lower()}, "
                    "which is not supported for SerialManipulator conversion. "
                    "Select a different tip or exclude planar/floating joints."
                )
            # Get joint position in world frame
            parent_T = fk[joint.parent]
            joint_T = parent_T @ joint.origin.matrix

            # Joint axis in world frame
            w = joint_T[:3, :3] @ joint.axis
            w = w / np.linalg.norm(w)

            # Point on joint axis
            p = joint_T[:3, 3]

            # Compute screw axis
            if joint.joint_type in (JointType.REVOLUTE, JointType.CONTINUOUS):
                # Revolute: S = [w; -w x p]
                v = -np.cross(w, p)
                S_list[:3, i] = w
                S_list[3:, i] = v
            else:
                # Prismatic: S = [0; v]
                S_list[:3, i] = 0
                S_list[3:, i] = w

            omega_list[:, i] = w
            r_list[:, i] = p

            # Spatial inertia
            child_link = self._links[joint.child]
            if child_link.inertial:
                G_list.append(child_link.inertial.spatial_inertia)
            else:
                G_list.append(np.eye(6, dtype=np.float64))

        # Body-frame screw axes: B = Ad(M^-1) * S
        M_inv = np.linalg.inv(M)
        Ad_M_inv = adjoint_transform(M_inv)
        B_list = Ad_M_inv @ S_list

        return {
            "M": M,
            "S_list": S_list,
            "B_list": B_list,
            "G_list": G_list,
            "omega_list": omega_list,
            "r_list": r_list,
            "joint_limits": self.joint_limits,
        }

    def to_serial_manipulator(self, tip_link: Optional[str] = None) -> "SerialManipulator":
        """
        Convert to ManipulaPy SerialManipulator.

        Args:
            tip_link: Optional link name to use as end-effector. Defaults to first end link.

        Returns:
            SerialManipulator instance ready for IK/FK
        """
        from ..kinematics import SerialManipulator
        from ..utils import extract_omega_list

        params = self.extract_screw_axes(tip_link=tip_link)

        return SerialManipulator(
            M_list=params["M"],
            omega_list=extract_omega_list(params["S_list"]),
            S_list=params["S_list"],
            B_list=params["B_list"],
            G_list=params["G_list"],
            joint_limits=params["joint_limits"],
        )

    def to_manipulator_dynamics(self) -> "ManipulatorDynamics":
        """
        Convert to ManipulaPy ManipulatorDynamics.

        Returns:
            ManipulatorDynamics instance ready for dynamics computation
        """
        from ..dynamics import ManipulatorDynamics
        from ..utils import extract_omega_list, extract_r_list

        params = self.extract_screw_axes()

        return ManipulatorDynamics(
            M_list=params["M"],
            omega_list=extract_omega_list(params["S_list"]),
            r_list=extract_r_list(params["S_list"]),
            b_list=None,
            S_list=params["S_list"],
            B_list=params["B_list"],
            Glist=params["G_list"],
        )

    # ==================== Visualization ====================

    def show(
        self,
        cfg: Optional[Union[np.ndarray, Dict[str, float]]] = None,
        use_collision: bool = False,
    ) -> None:
        """
        Visualize the robot.

        Args:
            cfg: Joint configuration
            use_collision: Show collision geometry instead of visual
        """
        from .visualization import show_robot

        show_robot(self, cfg, use_collision)

    def animate(
        self,
        cfg_trajectory: Union[Dict[str, np.ndarray], np.ndarray],
        loop_time: float = 3.0,
        use_collision: bool = False,
    ) -> None:
        """
        Animate robot along trajectory.

        Args:
            cfg_trajectory: Joint trajectories {joint_name: array} or (N, DOF) array
            loop_time: Animation duration in seconds
            use_collision: Use collision geometry
        """
        from .visualization import animate_robot

        animate_robot(self, cfg_trajectory, loop_time, use_collision)

    # ==================== Utilities ====================

    def get_link(self, name: str) -> Optional[Link]:
        """Get link by name."""
        return self._links.get(name)

    def get_joint(self, name: str) -> Optional[Joint]:
        """Get joint by name."""
        return self._joints.get(name)

    def get_chain(self, root: str, tip: str) -> List[Joint]:
        """
        Get joint chain between two links.

        Args:
            root: Root link name
            tip: Tip link name

        Returns:
            List of joints from root to tip
        """
        # Build child->parent mapping
        child_to_joint = {j.child: j for j in self._joints.values()}

        chain = []
        current = tip

        while current != root:
            if current not in child_to_joint:
                raise ValueError(f"No path from {root} to {tip}")

            joint = child_to_joint[current]
            chain.append(joint)
            current = joint.parent

        return list(reversed(chain))

    def copy(self) -> "URDF":
        """Create a deep copy of the URDF."""
        import copy

        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return (
            f"URDF(name='{self.name}', "
            f"links={len(self._links)}, "
            f"joints={len(self._joints)}, "
            f"dofs={self.num_dofs})"
        )

    def __str__(self) -> str:
        lines = [
            f"URDF: {self.name}",
            f"  Links: {len(self._links)}",
            f"  Joints: {len(self._joints)}",
            f"  Actuated DOFs: {self.num_dofs}",
            f"  Root link: {self._root_link_name}",
            f"  End links: {self._end_link_names}",
        ]
        return "\n".join(lines)
