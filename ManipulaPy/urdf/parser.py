#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
URDF XML Parser

Robust XML parsing for URDF files. Handles malformed XML gracefully.

Copyright (c) 2025 Mohamed Aboelnasr
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import numpy as np

from .types import (
    Actuator,
    Box,
    Collision,
    Cylinder,
    Geometry,
    Inertial,
    Joint,
    JointCalibration,
    JointDynamics,
    JointLimit,
    JointMimic,
    JointType,
    Link,
    Material,
    Mesh,
    Origin,
    SafetyController,
    Sphere,
    Transmission,
    TransmissionJoint,
    Visual,
)

if TYPE_CHECKING:
    from .core import URDF

logger = logging.getLogger(__name__)


class URDFParser:
    """
    URDF XML parser with robust error handling.

    Handles:
    - Standard URDF XML files
    - Malformed XML (best-effort parsing)
    - package:// resource paths
    - Relative and absolute mesh paths
    """

    @classmethod
    def parse_file(
        cls,
        filename: Union[str, Path],
        load_meshes: bool = False,
        mesh_dir: Optional[Path] = None,
        package_map: Optional[Dict[str, Union[str, Path]]] = None,
    ) -> "URDF":
        """
        Parse URDF from file.

        Args:
            filename: Path to URDF file
            load_meshes: Whether to load mesh geometry
            mesh_dir: Base directory for mesh file resolution
            package_map: Optional mapping of package names to paths

        Returns:
            URDF object
        """
        from .core import URDF
        from .resolver import PackageResolver

        path = Path(filename).resolve()

        if not path.exists():
            raise FileNotFoundError(f"URDF file not found: {path}")

        # Handle xacro files
        if path.suffix.lower() in (".xacro", ".urdf.xacro") or ".xacro" in path.name:
            from .xacro import XacroProcessor

            xml_string = XacroProcessor.process(path)
        else:
            xml_string = path.read_text(encoding="utf-8")

        if mesh_dir is None:
            mesh_dir = path.parent

        # Create package resolver
        resolver = PackageResolver.for_urdf(path)
        if package_map:
            for name, pkg_path in package_map.items():
                resolver.add_package(name, pkg_path)
        if mesh_dir:
            resolver.add_search_path(mesh_dir)

        return cls.parse_string(
            xml_string,
            base_path=path.parent,
            mesh_dir=mesh_dir,
            load_meshes=load_meshes,
            filename_handler=resolver.create_handler(),
        )

    @classmethod
    def parse_string(
        cls,
        xml_string: str,
        base_path: Optional[Path] = None,
        mesh_dir: Optional[Path] = None,
        load_meshes: bool = False,
        filename_handler: Optional[Callable[[str], str]] = None,
    ) -> "URDF":
        """
        Parse URDF from XML string.

        Args:
            xml_string: URDF XML content
            base_path: Base path for relative file resolution
            mesh_dir: Directory for mesh files
            load_meshes: Whether to load mesh geometry
            filename_handler: Optional custom filename resolver

        Returns:
            URDF object
        """
        from .core import URDF

        # Try parsing XML
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as e:
            logger.warning(f"XML parse error, attempting recovery: {e}")
            root = cls._parse_with_recovery(xml_string)

        if root.tag != "robot":
            raise ValueError(f"Expected <robot> root element, got <{root.tag}>")

        name = root.get("name", "robot")

        # Create or use provided filename handler for mesh resolution
        if filename_handler is None:
            filename_handler = cls._create_filename_handler(base_path, mesh_dir)

        # Parse materials first (can be referenced by visuals)
        materials: Dict[str, Material] = {}
        for elem in root.findall("material"):
            mat = cls._parse_material(elem)
            if mat.name:
                materials[mat.name] = mat

        # Parse links
        links: List[Link] = []
        for elem in root.findall("link"):
            link = cls._parse_link(elem, materials, filename_handler, load_meshes)
            links.append(link)

        # Parse joints
        joints: List[Joint] = []
        for elem in root.findall("joint"):
            joint = cls._parse_joint(elem)
            joints.append(joint)

        # Parse transmissions
        transmissions: List[Transmission] = []
        for elem in root.findall("transmission"):
            transmission = cls._parse_transmission(elem)
            transmissions.append(transmission)

        return URDF(
            name=name,
            links=links,
            joints=joints,
            materials=materials,
            transmissions=transmissions,
            filename_handler=filename_handler,
        )

    @classmethod
    def _parse_with_recovery(cls, xml_string: str) -> ET.Element:
        """
        Attempt to parse malformed XML with recovery.

        Tries various recovery strategies for common XML issues, such as
        stripping a leading ``<?xml ...?>`` declaration and wrapping the
        content in a ``<robot>`` root element when one is missing.

        Args:
            xml_string: Raw URDF XML document text to parse.

        Returns:
            ET.Element: The parsed XML root element.

        Raises:
            ValueError: If parsing fails even after all recovery attempts.
        """
        # Try removing XML declaration issues
        lines = xml_string.split("\n")
        if lines and lines[0].startswith("<?xml"):
            lines = lines[1:]
        recovered = "\n".join(lines)

        try:
            return ET.fromstring(recovered)
        except ET.ParseError:
            pass

        # Try wrapping in robot tag if missing
        if "<robot" not in xml_string.lower():
            wrapped = f'<robot name="recovered">{xml_string}</robot>'
            try:
                return ET.fromstring(wrapped)
            except ET.ParseError:
                pass

        raise ValueError("Failed to parse URDF XML even with recovery attempts")

    @classmethod
    def _create_filename_handler(
        cls, base_path: Optional[Path], mesh_dir: Optional[Path]
    ) -> Callable[[str], str]:
        """Create a filename resolution handler.

        Builds a closure that resolves mesh filenames against the provided
        directories, expanding ``package://`` and ``file://`` URIs and
        resolving relative paths.

        Args:
            base_path: Directory of the URDF file, used as a fallback root
                for resolving relative and package-relative paths. May be None.
            mesh_dir: Directory to search first for mesh files. May be None.

        Returns:
            Callable[[str], str]: A handler that maps a raw URDF filename to a
            resolved filesystem path, returning the input unchanged if it
            cannot be resolved.
        """

        def handler(filename: str) -> str:
            """Resolve a mesh filename, expanding package:// and file:// URIs.

            Args:
                filename: Raw filename from a URDF mesh element, possibly a
                    ``package://`` or ``file://`` URI or a relative path.

            Returns:
                str: The resolved filesystem path, or the original filename
                unchanged when it is empty or cannot be resolved.
            """
            if not filename:
                return filename

            # Handle package:// URIs
            if filename.startswith("package://"):
                # Extract package path
                path_parts = filename[10:].split("/", 1)
                if len(path_parts) == 2:
                    package_name, rel_path = path_parts
                    # Try to find in mesh_dir or base_path
                    if mesh_dir:
                        candidate = mesh_dir / rel_path
                        if candidate.exists():
                            return str(candidate)
                        # Try with package name as subdirectory
                        candidate = mesh_dir / package_name / rel_path
                        if candidate.exists():
                            return str(candidate)
                    if base_path:
                        candidate = base_path / rel_path
                        if candidate.exists():
                            return str(candidate)
                # Return original if not resolved
                return filename

            # Handle file:// URIs
            if filename.startswith("file://"):
                return filename[7:]

            # Handle relative paths
            if not Path(filename).is_absolute():
                if mesh_dir:
                    candidate = mesh_dir / filename
                    if candidate.exists():
                        return str(candidate)
                if base_path:
                    candidate = base_path / filename
                    if candidate.exists():
                        return str(candidate)

            return filename

        return handler

    # ==================== Element Parsers ====================

    @classmethod
    def _parse_origin(cls, elem: Optional[ET.Element]) -> Origin:
        """Parse an ``<origin>`` element into an Origin.

        Args:
            elem: The ``<origin>`` XML element, or None. When None (or when its
                ``xyz``/``rpy`` attributes fail to parse) a default Origin is
                returned.

        Returns:
            Origin: Origin with ``xyz`` translation (meters) and ``rpy`` Euler
            angles (radians) parsed from the element's attributes.
        """
        if elem is None:
            return Origin()

        xyz_str = elem.get("xyz", "0 0 0")
        rpy_str = elem.get("rpy", "0 0 0")

        try:
            xyz = np.array([float(x) for x in xyz_str.split()], dtype=np.float64)
            rpy = np.array([float(x) for x in rpy_str.split()], dtype=np.float64)
        except (ValueError, TypeError):
            logger.warning(f"Failed to parse origin: xyz={xyz_str}, rpy={rpy_str}")
            return Origin()

        return Origin(xyz=xyz, rpy=rpy)

    @classmethod
    def _parse_inertia_matrix(cls, elem: Optional[ET.Element]) -> np.ndarray:
        """Parse an ``<inertia>`` element into a 3x3 inertia matrix.

        Args:
            elem: The ``<inertia>`` XML element carrying the ``ixx``, ``ixy``,
                ``ixz``, ``iyy``, ``iyz`` and ``izz`` attributes, or None.

        Returns:
            np.ndarray: Symmetric (3, 3) float64 inertia tensor; all zeros when
            ``elem`` is None.
        """
        if elem is None:
            return np.zeros((3, 3), dtype=np.float64)

        ixx = float(elem.get("ixx", 0))
        ixy = float(elem.get("ixy", 0))
        ixz = float(elem.get("ixz", 0))
        iyy = float(elem.get("iyy", 0))
        iyz = float(elem.get("iyz", 0))
        izz = float(elem.get("izz", 0))

        return np.array(
            [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=np.float64
        )

    @classmethod
    def _parse_inertial(cls, elem: Optional[ET.Element]) -> Optional[Inertial]:
        """Parse an ``<inertial>`` element into an Inertial.

        Args:
            elem: The ``<inertial>`` XML element containing optional
                ``<origin>``, ``<mass>`` and ``<inertia>`` children, or None.

        Returns:
            Optional[Inertial]: Inertial with mass (kg), origin and inertia
            tensor; None when ``elem`` is None.
        """
        if elem is None:
            return None

        origin = cls._parse_origin(elem.find("origin"))

        mass_elem = elem.find("mass")
        mass = float(mass_elem.get("value", 0)) if mass_elem is not None else 0.0

        inertia = cls._parse_inertia_matrix(elem.find("inertia"))

        return Inertial(mass=mass, origin=origin, inertia=inertia)

    @classmethod
    def _parse_geometry(
        cls,
        elem: Optional[ET.Element],
        filename_handler: Optional[Callable[[str], str]] = None,
        load_mesh: bool = False,
    ) -> Optional[Geometry]:
        """Parse a ``<geometry>`` element into a geometry primitive.

        Args:
            elem: The ``<geometry>`` XML element wrapping a ``<box>``,
                ``<cylinder>``, ``<sphere>`` or ``<mesh>`` child, or None.
            filename_handler: Optional callable used to resolve a mesh
                filename to a filesystem path.
            load_mesh: If True, eagerly load the mesh geometry data after
                resolving its filename.

        Returns:
            Optional[Geometry]: A Box, Cylinder, Sphere or Mesh instance, or
            None when ``elem`` is None or contains no recognized child.
        """
        if elem is None:
            return None

        # Box
        box_elem = elem.find("box")
        if box_elem is not None:
            size_str = box_elem.get("size", "1 1 1")
            size = np.array([float(x) for x in size_str.split()], dtype=np.float64)
            return Box(size=size)

        # Cylinder
        cyl_elem = elem.find("cylinder")
        if cyl_elem is not None:
            return Cylinder(
                radius=float(cyl_elem.get("radius", 1)),
                length=float(cyl_elem.get("length", 1)),
            )

        # Sphere
        sphere_elem = elem.find("sphere")
        if sphere_elem is not None:
            return Sphere(radius=float(sphere_elem.get("radius", 1)))

        # Mesh
        mesh_elem = elem.find("mesh")
        if mesh_elem is not None:
            filename = mesh_elem.get("filename", "")

            # Resolve filename
            if filename_handler:
                filename = filename_handler(filename)

            # Parse scale
            scale_str = mesh_elem.get("scale", "1 1 1")
            scale_parts = scale_str.split()
            if len(scale_parts) == 1:
                scale = np.full(3, float(scale_parts[0]), dtype=np.float64)
            else:
                scale = np.array([float(x) for x in scale_parts], dtype=np.float64)

            mesh = Mesh(filename=filename, scale=scale)

            # Optionally load mesh data
            if load_mesh and filename:
                mesh._load_mesh()

            return mesh

        return None

    @classmethod
    def _parse_material(cls, elem: Optional[ET.Element]) -> Material:
        """Parse a ``<material>`` element into a Material.

        Args:
            elem: The ``<material>`` XML element with an optional ``name``
                attribute and ``<color>``/``<texture>`` children, or None.

        Returns:
            Material: Material with name, optional RGBA color (length-4 float64
            array) and optional texture filename; a default Material when
            ``elem`` is None.
        """
        if elem is None:
            return Material()

        name = elem.get("name", "")

        color = None
        color_elem = elem.find("color")
        if color_elem is not None:
            rgba_str = color_elem.get("rgba", "1 1 1 1")
            color = np.array([float(x) for x in rgba_str.split()], dtype=np.float64)

        texture = None
        texture_elem = elem.find("texture")
        if texture_elem is not None:
            texture = texture_elem.get("filename")

        return Material(name=name, color=color, texture=texture)

    @classmethod
    def _parse_visual(
        cls,
        elem: ET.Element,
        materials: Dict[str, Material],
        filename_handler: Optional[Callable[[str], str]] = None,
        load_mesh: bool = False,
    ) -> Visual:
        """Parse a ``<visual>`` element into a Visual.

        Args:
            elem: The ``<visual>`` XML element with optional ``<origin>``,
                ``<geometry>`` and ``<material>`` children.
            materials: Mapping of material name to Material used to resolve a
                visual's referenced material by name.
            filename_handler: Optional callable used to resolve mesh filenames
                to filesystem paths.
            load_mesh: If True, eagerly load mesh geometry data.

        Returns:
            Visual: Visual with name, origin, geometry and resolved material.
        """
        name = elem.get("name", "")
        origin = cls._parse_origin(elem.find("origin"))
        geometry = cls._parse_geometry(
            elem.find("geometry"), filename_handler, load_mesh
        )

        material = None
        mat_elem = elem.find("material")
        if mat_elem is not None:
            mat_name = mat_elem.get("name", "")
            if mat_name in materials:
                material = materials[mat_name]
            else:
                material = cls._parse_material(mat_elem)

        return Visual(name=name, origin=origin, geometry=geometry, material=material)

    @classmethod
    def _parse_collision(
        cls,
        elem: ET.Element,
        filename_handler: Optional[Callable[[str], str]] = None,
        load_mesh: bool = False,
    ) -> Collision:
        """Parse a ``<collision>`` element into a Collision.

        Args:
            elem: The ``<collision>`` XML element with optional ``<origin>``
                and ``<geometry>`` children.
            filename_handler: Optional callable used to resolve mesh filenames
                to filesystem paths.
            load_mesh: If True, eagerly load mesh geometry data.

        Returns:
            Collision: Collision with name, origin and geometry.
        """
        name = elem.get("name", "")
        origin = cls._parse_origin(elem.find("origin"))
        geometry = cls._parse_geometry(
            elem.find("geometry"), filename_handler, load_mesh
        )

        return Collision(name=name, origin=origin, geometry=geometry)

    @classmethod
    def _parse_link(
        cls,
        elem: ET.Element,
        materials: Dict[str, Material],
        filename_handler: Optional[Callable[[str], str]] = None,
        load_meshes: bool = False,
    ) -> Link:
        """Parse a ``<link>`` element into a Link.

        Args:
            elem: The ``<link>`` XML element, which must carry a ``name``
                attribute and may contain ``<inertial>``, ``<visual>`` and
                ``<collision>`` children.
            materials: Mapping of material name to Material used to resolve
                materials referenced by the link's visuals.
            filename_handler: Optional callable used to resolve mesh filenames
                to filesystem paths.
            load_meshes: If True, eagerly load mesh geometry data for visuals
                and collisions.

        Returns:
            Link: Link with name, inertial properties, visuals and collisions.

        Raises:
            ValueError: If the element has no ``name`` attribute.
        """
        name = elem.get("name", "")

        if not name:
            raise ValueError("Link missing required 'name' attribute")

        inertial = cls._parse_inertial(elem.find("inertial"))

        visuals = [
            cls._parse_visual(v, materials, filename_handler, load_meshes)
            for v in elem.findall("visual")
        ]

        collisions = [
            cls._parse_collision(c, filename_handler, load_meshes)
            for c in elem.findall("collision")
        ]

        return Link(
            name=name, inertial=inertial, visuals=visuals, collisions=collisions
        )

    @classmethod
    def _parse_joint_limit(cls, elem: Optional[ET.Element]) -> Optional[JointLimit]:
        """Parse a ``<limit>`` element into a JointLimit.

        Args:
            elem: The ``<limit>`` XML element with ``lower``, ``upper``,
                ``effort`` and ``velocity`` attributes, or None.

        Returns:
            Optional[JointLimit]: Joint limit with lower/upper position bounds
            (radians or meters), effort (N or Nm) and velocity limits; None
            when ``elem`` is None.
        """
        if elem is None:
            return None

        return JointLimit(
            lower=float(elem.get("lower", 0)),
            upper=float(elem.get("upper", 0)),
            effort=float(elem.get("effort", 0)),
            velocity=float(elem.get("velocity", 0)),
        )

    @classmethod
    def _parse_joint_dynamics(
        cls, elem: Optional[ET.Element]
    ) -> Optional[JointDynamics]:
        """Parse a ``<dynamics>`` element into a JointDynamics.

        Args:
            elem: The ``<dynamics>`` XML element with ``damping`` and
                ``friction`` attributes, or None.

        Returns:
            Optional[JointDynamics]: Joint dynamics with damping and friction
            coefficients; None when ``elem`` is None.
        """
        if elem is None:
            return None

        return JointDynamics(
            damping=float(elem.get("damping", 0)),
            friction=float(elem.get("friction", 0)),
        )

    @classmethod
    def _parse_joint_mimic(cls, elem: Optional[ET.Element]) -> Optional[JointMimic]:
        """Parse a ``<mimic>`` element into a JointMimic.

        Args:
            elem: The ``<mimic>`` XML element with ``joint``, ``multiplier``
                and ``offset`` attributes, or None.

        Returns:
            Optional[JointMimic]: Mimic spec naming the mimicked joint plus its
            multiplier and offset; None when ``elem`` is None.
        """
        if elem is None:
            return None

        return JointMimic(
            joint=elem.get("joint", ""),
            multiplier=float(elem.get("multiplier", 1)),
            offset=float(elem.get("offset", 0)),
        )

    @classmethod
    def _parse_safety_controller(
        cls, elem: Optional[ET.Element]
    ) -> Optional[SafetyController]:
        """Parse a ``<safety_controller>`` element into a SafetyController.

        Args:
            elem: The ``<safety_controller>`` XML element with
                ``soft_lower_limit``, ``soft_upper_limit``, ``k_position`` and
                ``k_velocity`` attributes, or None.

        Returns:
            Optional[SafetyController]: Safety controller with soft position
            limits and position/velocity gains; None when ``elem`` is None.
        """
        if elem is None:
            return None

        return SafetyController(
            soft_lower_limit=float(elem.get("soft_lower_limit", 0)),
            soft_upper_limit=float(elem.get("soft_upper_limit", 0)),
            k_position=float(elem.get("k_position", 0)),
            k_velocity=float(elem.get("k_velocity", 0)),
        )

    @classmethod
    def _parse_joint_calibration(
        cls, elem: Optional[ET.Element]
    ) -> Optional[JointCalibration]:
        """Parse a ``<calibration>`` element into a JointCalibration.

        Args:
            elem: The ``<calibration>`` XML element with optional ``rising``
                and ``falling`` attributes, or None.

        Returns:
            Optional[JointCalibration]: Calibration with rising/falling
            reference positions (each None if its attribute is absent); None
            when ``elem`` is None.
        """
        if elem is None:
            return None

        rising = elem.get("rising")
        falling = elem.get("falling")

        return JointCalibration(
            rising=float(rising) if rising is not None else None,
            falling=float(falling) if falling is not None else None,
        )

    @classmethod
    def _parse_joint(cls, elem: ET.Element) -> Joint:
        """Parse a ``<joint>`` element into a Joint.

        Args:
            elem: The ``<joint>`` XML element, which must carry ``name`` and
                ``type`` attributes and ``<parent>``/``<child>`` children, and
                may contain ``<origin>``, ``<axis>``, ``<limit>``,
                ``<dynamics>``, ``<mimic>``, ``<safety_controller>`` and
                ``<calibration>`` children.

        Returns:
            Joint: Joint with type, parent/child link names, origin, a
            unit-normalized rotation/translation axis (length-3 float64 array)
            and optional limit, dynamics, mimic, safety and calibration data.

        Raises:
            ValueError: If the joint has no ``name`` attribute, is missing a
                ``<parent>`` or ``<child>`` element, or those elements lack a
                ``link`` attribute.
        """
        name = elem.get("name", "")
        type_str = elem.get("type", "fixed")

        if not name:
            raise ValueError("Joint missing required 'name' attribute")

        joint_type = JointType.from_string(type_str)

        parent_elem = elem.find("parent")
        child_elem = elem.find("child")

        if parent_elem is None or child_elem is None:
            raise ValueError(f"Joint '{name}' missing parent or child element")

        parent = parent_elem.get("link", "")
        child = child_elem.get("link", "")

        if not parent or not child:
            raise ValueError(f"Joint '{name}' missing parent or child link attribute")

        origin = cls._parse_origin(elem.find("origin"))

        # Parse axis
        axis_elem = elem.find("axis")
        if axis_elem is not None:
            axis_str = axis_elem.get("xyz", "1 0 0")
            axis = np.array([float(x) for x in axis_str.split()], dtype=np.float64)
        else:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        # Normalize axis
        norm = np.linalg.norm(axis)
        if norm > 1e-10:
            axis = axis / norm

        limit = cls._parse_joint_limit(elem.find("limit"))
        dynamics = cls._parse_joint_dynamics(elem.find("dynamics"))
        mimic = cls._parse_joint_mimic(elem.find("mimic"))
        safety = cls._parse_safety_controller(elem.find("safety_controller"))
        calibration = cls._parse_joint_calibration(elem.find("calibration"))

        return Joint(
            name=name,
            joint_type=joint_type,
            parent=parent,
            child=child,
            origin=origin,
            axis=axis,
            limit=limit,
            dynamics=dynamics,
            mimic=mimic,
            safety=safety,
            calibration=calibration,
        )

    @classmethod
    def _parse_transmission(cls, elem: ET.Element) -> Transmission:
        """Parse a ``<transmission>`` element into a Transmission.

        The transmission type may be given either as a ``<type>`` child element
        or as a ``type`` attribute. Joints without a ``name`` and actuators
        without a ``name`` are skipped.

        Args:
            elem: The ``<transmission>`` XML element with an optional ``name``
                attribute, an optional type, and ``<joint>``/``<actuator>``
                children.

        Returns:
            Transmission: Transmission with name, type, and its list of
            transmission joints (with hardware interfaces) and actuators (with
            mechanical reduction and hardware interfaces).
        """
        name = elem.get("name", "")

        # Get transmission type
        # Can be in <type> element or as attribute
        type_elem = elem.find("type")
        if type_elem is not None and type_elem.text:
            trans_type = type_elem.text.strip()
        else:
            trans_type = elem.get("type", "")

        # Parse joints within transmission
        joints: List[TransmissionJoint] = []
        for joint_elem in elem.findall("joint"):
            joint_name = joint_elem.get("name", "")
            if not joint_name:
                continue

            # Hardware interface can be in <hardwareInterface> element
            hw_interface = None
            hw_elem = joint_elem.find("hardwareInterface")
            if hw_elem is not None and hw_elem.text:
                hw_interface = hw_elem.text.strip()

            joints.append(
                TransmissionJoint(
                    name=joint_name,
                    hardware_interface=hw_interface,
                )
            )

        # Parse actuators within transmission
        actuators: List[Actuator] = []
        for act_elem in elem.findall("actuator"):
            act_name = act_elem.get("name", "")
            if not act_name:
                continue

            # Mechanical reduction
            mech_red = 1.0
            mech_elem = act_elem.find("mechanicalReduction")
            if mech_elem is not None and mech_elem.text:
                try:
                    mech_red = float(mech_elem.text.strip())
                except ValueError:
                    pass

            # Hardware interface
            hw_interface = None
            hw_elem = act_elem.find("hardwareInterface")
            if hw_elem is not None and hw_elem.text:
                hw_interface = hw_elem.text.strip()

            actuators.append(
                Actuator(
                    name=act_name,
                    mechanical_reduction=mech_red,
                    hardware_interface=hw_interface,
                )
            )

        return Transmission(
            name=name,
            type=trans_type,
            joints=joints,
            actuators=actuators,
        )
