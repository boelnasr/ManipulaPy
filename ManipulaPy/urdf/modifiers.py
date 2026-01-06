#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
URDF Modifiers Module

Provides tools for programmatic modification of URDF models,
including calibration, payload simulation, and parameter adjustment.

Copyright (c) 2025 Mohamed Aboelnasr
"""

import numpy as np
from copy import deepcopy
from typing import Dict, List, Optional, Union
from pathlib import Path
import json
import logging

from .types import (
    Link, Joint, JointType, Origin, Inertial, JointLimit,
)

logger = logging.getLogger(__name__)


class URDFModifier:
    """
    Modify URDF models programmatically.

    Supports joint calibration, link property changes, payload addition,
    and model composition. All operations create a deep copy of the URDF
    to avoid modifying the original.

    Example:
        >>> modifier = URDFModifier(urdf)
        >>> modifier.set_joint_origin("joint1", xyz=[0, 0, 0.1])
        >>> modifier.offset_joint_zero("joint2", offset=0.01)
        >>> modifier.add_payload("ee_link", mass=2.0)
        >>> calibrated_urdf = modifier.urdf
    """

    def __init__(self, urdf: "URDF"):
        """
        Create modifier for URDF.

        Args:
            urdf: URDF model to modify (will be deep-copied)
        """
        self._urdf = deepcopy(urdf)

    @property
    def urdf(self) -> "URDF":
        """Get the modified URDF."""
        return self._urdf

    # ==================== Joint Modifications ====================

    def set_joint_origin(
        self,
        joint_name: str,
        xyz: Optional[np.ndarray] = None,
        rpy: Optional[np.ndarray] = None,
    ) -> "URDFModifier":
        """
        Modify joint origin (for calibration).

        Args:
            joint_name: Name of joint to modify
            xyz: New position [x, y, z] or None to keep current
            rpy: New orientation [roll, pitch, yaw] or None to keep current

        Returns:
            self for method chaining
        """
        if joint_name not in self._urdf._joints:
            raise ValueError(f"Joint '{joint_name}' not found")

        joint = self._urdf._joints[joint_name]

        if xyz is not None:
            joint.origin.xyz = np.asarray(xyz, dtype=np.float64)
            joint.origin._matrix_cache = None  # Clear cache

        if rpy is not None:
            joint.origin.rpy = np.asarray(rpy, dtype=np.float64)
            joint.origin._matrix_cache = None  # Clear cache

        return self

    def set_joint_axis(
        self,
        joint_name: str,
        axis: np.ndarray,
    ) -> "URDFModifier":
        """
        Modify joint axis direction.

        Args:
            joint_name: Name of joint to modify
            axis: New axis direction [x, y, z] (will be normalized)

        Returns:
            self for method chaining
        """
        if joint_name not in self._urdf._joints:
            raise ValueError(f"Joint '{joint_name}' not found")

        joint = self._urdf._joints[joint_name]
        axis = np.asarray(axis, dtype=np.float64)
        norm = np.linalg.norm(axis)
        if norm > 1e-10:
            joint.axis = axis / norm
        else:
            raise ValueError("Axis cannot be zero vector")

        return self

    def set_joint_limits(
        self,
        joint_name: str,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        velocity: Optional[float] = None,
        effort: Optional[float] = None,
    ) -> "URDFModifier":
        """
        Modify joint limits.

        Args:
            joint_name: Name of joint to modify
            lower: New lower limit (rad or m)
            upper: New upper limit (rad or m)
            velocity: New velocity limit
            effort: New effort limit

        Returns:
            self for method chaining
        """
        if joint_name not in self._urdf._joints:
            raise ValueError(f"Joint '{joint_name}' not found")

        joint = self._urdf._joints[joint_name]

        if joint.limit is None:
            joint.limit = JointLimit()

        if lower is not None:
            joint.limit.lower = lower
        if upper is not None:
            joint.limit.upper = upper
        if velocity is not None:
            joint.limit.velocity = velocity
        if effort is not None:
            joint.limit.effort = effort

        return self

    def offset_joint_zero(
        self,
        joint_name: str,
        offset: float,
    ) -> "URDFModifier":
        """
        Add offset to joint zero position (calibration).

        For revolute joints, rotates the origin by the offset angle.
        For prismatic joints, translates the origin by the offset distance.

        Args:
            joint_name: Name of joint to calibrate
            offset: Offset angle (rad) or distance (m)

        Returns:
            self for method chaining
        """
        if joint_name not in self._urdf._joints:
            raise ValueError(f"Joint '{joint_name}' not found")

        joint = self._urdf._joints[joint_name]

        if joint.joint_type in (JointType.REVOLUTE, JointType.CONTINUOUS):
            # Apply rotation offset around the joint axis
            # This modifies the origin to shift the zero position
            from scipy.spatial.transform import Rotation

            # Get current orientation
            current_rpy = joint.origin.rpy

            # Create rotation offset around joint axis
            R_offset = Rotation.from_rotvec(offset * joint.axis)
            R_current = Rotation.from_euler('xyz', current_rpy)
            R_new = R_offset * R_current

            joint.origin.rpy = R_new.as_euler('xyz')
            joint.origin._matrix_cache = None

        elif joint.joint_type == JointType.PRISMATIC:
            # Translate origin along axis
            joint.origin.xyz = joint.origin.xyz + offset * joint.axis
            joint.origin._matrix_cache = None

        return self

    # ==================== Link Modifications ====================

    def set_link_mass(
        self,
        link_name: str,
        mass: float,
    ) -> "URDFModifier":
        """
        Modify link mass.

        Args:
            link_name: Name of link to modify
            mass: New mass (kg)

        Returns:
            self for method chaining
        """
        if link_name not in self._urdf._links:
            raise ValueError(f"Link '{link_name}' not found")

        link = self._urdf._links[link_name]

        if link.inertial is None:
            link.inertial = Inertial(mass=mass)
        else:
            link.inertial.mass = mass

        return self

    def set_link_inertia(
        self,
        link_name: str,
        inertia: np.ndarray,
    ) -> "URDFModifier":
        """
        Modify link inertia matrix.

        Args:
            link_name: Name of link to modify
            inertia: 3x3 inertia matrix

        Returns:
            self for method chaining
        """
        if link_name not in self._urdf._links:
            raise ValueError(f"Link '{link_name}' not found")

        link = self._urdf._links[link_name]

        if link.inertial is None:
            link.inertial = Inertial(mass=1.0)

        link.inertial.inertia = np.asarray(inertia, dtype=np.float64)

        return self

    def set_link_com(
        self,
        link_name: str,
        com: np.ndarray,
    ) -> "URDFModifier":
        """
        Modify link center of mass position.

        Args:
            link_name: Name of link to modify
            com: New COM position [x, y, z] in link frame

        Returns:
            self for method chaining
        """
        if link_name not in self._urdf._links:
            raise ValueError(f"Link '{link_name}' not found")

        link = self._urdf._links[link_name]

        if link.inertial is None:
            link.inertial = Inertial(mass=1.0)

        link.inertial.origin.xyz = np.asarray(com, dtype=np.float64)
        link.inertial.origin._matrix_cache = None

        return self

    # ==================== Batch Modifications ====================

    def apply_calibration(
        self,
        calibration: Dict[str, Dict[str, float]],
    ) -> "URDFModifier":
        """
        Apply calibration data to multiple joints.

        Args:
            calibration: {joint_name: {"offset": float, "lower": float, "upper": float, ...}}

        Returns:
            self for method chaining

        Example:
            modifier.apply_calibration({
                "joint1": {"offset": 0.01},
                "joint2": {"offset": -0.005, "lower": -1.5, "upper": 1.5},
            })
        """
        for joint_name, params in calibration.items():
            if joint_name not in self._urdf._joints:
                logger.warning(f"Calibration: joint '{joint_name}' not found, skipping")
                continue

            if "offset" in params:
                self.offset_joint_zero(joint_name, params["offset"])

            if "lower" in params or "upper" in params:
                self.set_joint_limits(
                    joint_name,
                    lower=params.get("lower"),
                    upper=params.get("upper"),
                )

            if "xyz" in params:
                self.set_joint_origin(joint_name, xyz=params["xyz"])

            if "rpy" in params:
                self.set_joint_origin(joint_name, rpy=params["rpy"])

        return self

    def scale_masses(
        self,
        scale: float,
        link_names: Optional[List[str]] = None,
    ) -> "URDFModifier":
        """
        Scale masses of links (for payload simulation or uncertainty analysis).

        Args:
            scale: Mass scale factor
            link_names: Links to scale (None = all links with inertials)

        Returns:
            self for method chaining
        """
        links = link_names or list(self._urdf._links.keys())

        for name in links:
            if name not in self._urdf._links:
                continue

            link = self._urdf._links[name]
            if link.inertial is not None:
                link.inertial.mass *= scale
                link.inertial.inertia *= scale

        return self

    # ==================== Payload Operations ====================

    def add_payload(
        self,
        link_name: str,
        mass: float,
        com: Optional[np.ndarray] = None,
        inertia: Optional[np.ndarray] = None,
    ) -> "URDFModifier":
        """
        Add payload mass to a link.

        Combines the payload with existing link inertial properties
        using proper mass combination formulas.

        Args:
            link_name: Link to add payload to
            mass: Payload mass (kg)
            com: Payload COM in link frame (default: link COM)
            inertia: Payload inertia 3x3 matrix (default: point mass at COM)

        Returns:
            self for method chaining
        """
        if link_name not in self._urdf._links:
            raise ValueError(f"Link '{link_name}' not found")

        link = self._urdf._links[link_name]

        if link.inertial is None:
            # No existing inertial - create new one
            link.inertial = Inertial(mass=mass)
            if com is not None:
                link.inertial.origin.xyz = np.asarray(com, dtype=np.float64)
            if inertia is not None:
                link.inertial.inertia = np.asarray(inertia, dtype=np.float64)
        else:
            # Combine with existing inertial
            m1 = link.inertial.mass
            m2 = mass
            c1 = link.inertial.origin.xyz
            c2 = np.asarray(com, dtype=np.float64) if com is not None else c1

            # New total mass
            m_total = m1 + m2

            # New COM (weighted average)
            c_new = (m1 * c1 + m2 * c2) / m_total

            # Update mass and COM
            link.inertial.mass = m_total
            link.inertial.origin.xyz = c_new
            link.inertial.origin._matrix_cache = None

            # Add payload inertia (simplified - assumes both at new COM)
            if inertia is not None:
                link.inertial.inertia = link.inertial.inertia + np.asarray(inertia)

        return self

    def remove_link_inertial(self, link_name: str) -> "URDFModifier":
        """
        Remove inertial properties from a link.

        Args:
            link_name: Link to modify

        Returns:
            self for method chaining
        """
        if link_name not in self._urdf._links:
            raise ValueError(f"Link '{link_name}' not found")

        self._urdf._links[link_name].inertial = None
        return self

    # ==================== Structural Operations ====================

    def rename_joint(self, old_name: str, new_name: str) -> "URDFModifier":
        """
        Rename a joint.

        Args:
            old_name: Current joint name
            new_name: New joint name

        Returns:
            self for method chaining
        """
        if old_name not in self._urdf._joints:
            raise ValueError(f"Joint '{old_name}' not found")
        if new_name in self._urdf._joints:
            raise ValueError(f"Joint '{new_name}' already exists")

        joint = self._urdf._joints.pop(old_name)
        joint.name = new_name
        self._urdf._joints[new_name] = joint

        # Update mimic references
        for j in self._urdf._joints.values():
            if j.mimic and j.mimic.joint == old_name:
                j.mimic.joint = new_name

        return self

    def rename_link(self, old_name: str, new_name: str) -> "URDFModifier":
        """
        Rename a link.

        Args:
            old_name: Current link name
            new_name: New link name

        Returns:
            self for method chaining
        """
        if old_name not in self._urdf._links:
            raise ValueError(f"Link '{old_name}' not found")
        if new_name in self._urdf._links:
            raise ValueError(f"Link '{new_name}' already exists")

        link = self._urdf._links.pop(old_name)
        link.name = new_name
        self._urdf._links[new_name] = link

        # Update joint references
        for joint in self._urdf._joints.values():
            if joint.parent == old_name:
                joint.parent = new_name
            if joint.child == old_name:
                joint.child = new_name

        # Update cached names
        if self._urdf._root_link_name == old_name:
            self._urdf._root_link_name = new_name
        if self._urdf._end_link_names:
            self._urdf._end_link_names = [
                new_name if n == old_name else n
                for n in self._urdf._end_link_names
            ]

        return self

    # ==================== Export ====================

    def to_urdf_string(self) -> str:
        """
        Export modified URDF as XML string.

        Returns:
            URDF XML string
        """
        import xml.etree.ElementTree as ET

        root = ET.Element("robot", name=self._urdf.name)

        # Export materials
        for mat in self._urdf._materials.values():
            mat_elem = ET.SubElement(root, "material", name=mat.name)
            if mat.color is not None:
                ET.SubElement(
                    mat_elem, "color",
                    rgba=" ".join(f"{x:.6g}" for x in mat.color)
                )
            if mat.texture is not None:
                ET.SubElement(mat_elem, "texture", filename=mat.texture)

        # Export links
        for link in self._urdf._links.values():
            link_elem = ET.SubElement(root, "link", name=link.name)

            # Inertial
            if link.inertial is not None:
                inertial_elem = ET.SubElement(link_elem, "inertial")
                self._export_origin(inertial_elem, link.inertial.origin)
                ET.SubElement(
                    inertial_elem, "mass",
                    value=f"{link.inertial.mass:.6g}"
                )
                I = link.inertial.inertia
                ET.SubElement(
                    inertial_elem, "inertia",
                    ixx=f"{I[0,0]:.6g}", ixy=f"{I[0,1]:.6g}", ixz=f"{I[0,2]:.6g}",
                    iyy=f"{I[1,1]:.6g}", iyz=f"{I[1,2]:.6g}", izz=f"{I[2,2]:.6g}"
                )

            # Visuals
            for visual in link.visuals:
                self._export_visual(link_elem, visual)

            # Collisions
            for collision in link.collisions:
                self._export_collision(link_elem, collision)

        # Export joints
        for joint in self._urdf._joints.values():
            joint_elem = ET.SubElement(
                root, "joint",
                name=joint.name,
                type=joint.joint_type.value
            )
            self._export_origin(joint_elem, joint.origin)
            ET.SubElement(joint_elem, "parent", link=joint.parent)
            ET.SubElement(joint_elem, "child", link=joint.child)
            ET.SubElement(
                joint_elem, "axis",
                xyz=" ".join(f"{x:.6g}" for x in joint.axis)
            )

            if joint.limit is not None:
                limit_attrs = {
                    "lower": f"{joint.limit.lower:.6g}",
                    "upper": f"{joint.limit.upper:.6g}",
                }
                if joint.limit.velocity:
                    limit_attrs["velocity"] = f"{joint.limit.velocity:.6g}"
                if joint.limit.effort:
                    limit_attrs["effort"] = f"{joint.limit.effort:.6g}"
                ET.SubElement(joint_elem, "limit", **limit_attrs)

            if joint.dynamics is not None:
                ET.SubElement(
                    joint_elem, "dynamics",
                    damping=f"{joint.dynamics.damping:.6g}",
                    friction=f"{joint.dynamics.friction:.6g}"
                )

            if joint.mimic is not None:
                ET.SubElement(
                    joint_elem, "mimic",
                    joint=joint.mimic.joint,
                    multiplier=f"{joint.mimic.multiplier:.6g}",
                    offset=f"{joint.mimic.offset:.6g}"
                )

        # Export transmissions
        for trans in self._urdf._transmissions.values():
            trans_elem = ET.SubElement(root, "transmission", name=trans.name)
            if trans.type:
                type_elem = ET.SubElement(trans_elem, "type")
                type_elem.text = trans.type

            for tj in trans.joints:
                tj_elem = ET.SubElement(trans_elem, "joint", name=tj.name)
                if tj.hardware_interface:
                    hw_elem = ET.SubElement(tj_elem, "hardwareInterface")
                    hw_elem.text = tj.hardware_interface

            for act in trans.actuators:
                act_elem = ET.SubElement(trans_elem, "actuator", name=act.name)
                if act.mechanical_reduction != 1.0:
                    mr_elem = ET.SubElement(act_elem, "mechanicalReduction")
                    mr_elem.text = f"{act.mechanical_reduction:.6g}"
                if act.hardware_interface:
                    hw_elem = ET.SubElement(act_elem, "hardwareInterface")
                    hw_elem.text = act.hardware_interface

        # Pretty print
        self._indent_xml(root)
        return ET.tostring(root, encoding="unicode")

    def _export_origin(self, parent, origin: Origin) -> None:
        """Export origin element."""
        import xml.etree.ElementTree as ET

        xyz = " ".join(f"{x:.6g}" for x in origin.xyz)
        rpy = " ".join(f"{x:.6g}" for x in origin.rpy)
        ET.SubElement(parent, "origin", xyz=xyz, rpy=rpy)

    def _export_visual(self, parent, visual) -> None:
        """Export visual element."""
        import xml.etree.ElementTree as ET
        from .types import Box, Cylinder, Sphere, Mesh

        vis_elem = ET.SubElement(parent, "visual")
        if visual.name:
            vis_elem.set("name", visual.name)

        self._export_origin(vis_elem, visual.origin)

        if visual.geometry is not None:
            geom_elem = ET.SubElement(vis_elem, "geometry")
            self._export_geometry(geom_elem, visual.geometry)

        if visual.material is not None:
            if visual.material.name:
                ET.SubElement(vis_elem, "material", name=visual.material.name)

    def _export_collision(self, parent, collision) -> None:
        """Export collision element."""
        import xml.etree.ElementTree as ET

        col_elem = ET.SubElement(parent, "collision")
        if collision.name:
            col_elem.set("name", collision.name)

        self._export_origin(col_elem, collision.origin)

        if collision.geometry is not None:
            geom_elem = ET.SubElement(col_elem, "geometry")
            self._export_geometry(geom_elem, collision.geometry)

    def _export_geometry(self, parent, geometry) -> None:
        """Export geometry element."""
        import xml.etree.ElementTree as ET
        from .types import Box, Cylinder, Sphere, Mesh

        if isinstance(geometry, Box):
            ET.SubElement(
                parent, "box",
                size=" ".join(f"{x:.6g}" for x in geometry.size)
            )
        elif isinstance(geometry, Cylinder):
            ET.SubElement(
                parent, "cylinder",
                radius=f"{geometry.radius:.6g}",
                length=f"{geometry.length:.6g}"
            )
        elif isinstance(geometry, Sphere):
            ET.SubElement(parent, "sphere", radius=f"{geometry.radius:.6g}")
        elif isinstance(geometry, Mesh):
            attrs = {"filename": geometry.filename}
            if not np.allclose(geometry.scale, [1, 1, 1]):
                attrs["scale"] = " ".join(f"{x:.6g}" for x in geometry.scale)
            ET.SubElement(parent, "mesh", **attrs)

    def _indent_xml(self, elem, level: int = 0) -> None:
        """Add indentation to XML element."""
        indent = "\n" + "  " * level
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
            for child in elem:
                self._indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent

    def save(self, filename: Union[str, Path]) -> None:
        """
        Save modified URDF to file.

        Args:
            filename: Output file path
        """
        with open(filename, 'w') as f:
            f.write(self.to_urdf_string())


# ==================== Calibration File Support ====================


def load_calibration(filename: Union[str, Path]) -> Dict[str, Dict[str, float]]:
    """
    Load calibration data from file.

    Supports YAML and JSON formats.

    Args:
        filename: Path to calibration file

    Returns:
        Calibration dictionary

    Example YAML:
        joints:
          joint1:
            offset: 0.01
          joint2:
            offset: -0.005
            lower: -1.5
            upper: 1.5
    """
    path = Path(filename)

    if path.suffix in ('.yaml', '.yml'):
        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required for YAML calibration files: pip install pyyaml")
    elif path.suffix == '.json':
        with open(path) as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unknown calibration file format: {path.suffix}")

    return data.get('joints', data)


def save_calibration(
    calibration: Dict[str, Dict[str, float]],
    filename: Union[str, Path],
) -> None:
    """
    Save calibration data to file.

    Args:
        calibration: Calibration dictionary
        filename: Output file path
    """
    path = Path(filename)
    data = {'joints': calibration}

    if path.suffix in ('.yaml', '.yml'):
        try:
            import yaml
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        except ImportError:
            raise ImportError("PyYAML required for YAML calibration files: pip install pyyaml")
    elif path.suffix == '.json':
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError(f"Unknown calibration file format: {path.suffix}")
