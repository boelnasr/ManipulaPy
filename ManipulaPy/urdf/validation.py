#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
URDF Validation Module

Validates URDF structure for cycles, multiple roots, disconnected links,
and other common issues.

Copyright (c) 2025 Mohamed Aboelnasr
"""

import logging
from typing import List, Set, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    ERROR = "error"      # Invalid URDF, will fail to use
    WARNING = "warning"  # May cause unexpected behavior
    INFO = "info"        # Informational, not an issue


@dataclass
class ValidationIssue:
    """A validation issue found in the URDF."""

    severity: ValidationSeverity
    message: str
    element_type: str = ""  # "link", "joint", "transmission"
    element_name: str = ""  # Name of the problematic element

    def __str__(self) -> str:
        location = f" in {self.element_type} '{self.element_name}'" if self.element_name else ""
        return f"[{self.severity.value.upper()}]{location}: {self.message}"


@dataclass
class ValidationResult:
    """Result of URDF validation."""

    valid: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)

    def add_issue(
        self,
        severity: ValidationSeverity,
        message: str,
        element_type: str = "",
        element_name: str = "",
    ) -> None:
        """Add a validation issue."""
        issue = ValidationIssue(
            severity=severity,
            message=message,
            element_type=element_type,
            element_name=element_name,
        )
        self.issues.append(issue)

        if severity == ValidationSeverity.ERROR:
            self.valid = False

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get all error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get all warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def __str__(self) -> str:
        if not self.issues:
            return "URDF is valid"

        lines = [f"URDF validation {'FAILED' if not self.valid else 'passed with warnings'}:"]
        for issue in self.issues:
            lines.append(f"  {issue}")
        return "\n".join(lines)


class URDFValidator:
    """
    Validates URDF structure and semantics.

    Checks for:
    - Cycles in the kinematic tree
    - Multiple root links
    - Disconnected links
    - Missing link/joint references
    - Invalid joint configurations
    - Missing required elements

    Example:
        >>> validator = URDFValidator()
        >>> result = validator.validate(urdf)
        >>> if not result.valid:
        ...     print(result)
    """

    def validate(
        self,
        links: Dict[str, object],
        joints: Dict[str, object],
        transmissions: Optional[Dict[str, object]] = None,
    ) -> ValidationResult:
        """
        Validate URDF structure.

        Args:
            links: Dictionary of link name -> Link
            joints: Dictionary of joint name -> Joint
            transmissions: Optional dictionary of transmission name -> Transmission

        Returns:
            ValidationResult with all issues found
        """
        result = ValidationResult()

        # Check for empty URDF
        if not links:
            result.add_issue(
                ValidationSeverity.ERROR,
                "URDF has no links",
            )
            return result

        # Build parent-child graph
        child_to_parent: Dict[str, str] = {}  # child_link -> parent_link
        parent_to_children: Dict[str, List[str]] = {}  # parent_link -> [child_links]

        for joint_name, joint in joints.items():
            parent = joint.parent
            child = joint.child

            # Check for missing links
            if parent not in links:
                result.add_issue(
                    ValidationSeverity.ERROR,
                    f"Parent link '{parent}' not found",
                    "joint",
                    joint_name,
                )

            if child not in links:
                result.add_issue(
                    ValidationSeverity.ERROR,
                    f"Child link '{child}' not found",
                    "joint",
                    joint_name,
                )

            # Check for multiple parents
            if child in child_to_parent:
                result.add_issue(
                    ValidationSeverity.ERROR,
                    f"Link '{child}' has multiple parents: "
                    f"'{child_to_parent[child]}' and '{parent}'",
                    "joint",
                    joint_name,
                )
            else:
                child_to_parent[child] = parent

            # Build parent-to-children map
            if parent not in parent_to_children:
                parent_to_children[parent] = []
            parent_to_children[parent].append(child)

        # Find root links (links that are not children of any joint)
        all_children = set(child_to_parent.keys())
        root_links = [name for name in links.keys() if name not in all_children]

        if not root_links:
            result.add_issue(
                ValidationSeverity.ERROR,
                "No root link found - possible cycle in kinematic tree",
            )
        elif len(root_links) > 1:
            result.add_issue(
                ValidationSeverity.WARNING,
                f"Multiple root links found: {root_links}. "
                "Only the first will be used for kinematics.",
            )

        # Check for disconnected links
        if root_links:
            reachable = self._find_reachable_links(root_links[0], parent_to_children)
            disconnected = set(links.keys()) - reachable

            if disconnected:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    f"Disconnected links found: {sorted(disconnected)}",
                )

        # Check for cycles
        cycle = self._detect_cycle(links.keys(), child_to_parent)
        if cycle:
            result.add_issue(
                ValidationSeverity.ERROR,
                f"Cycle detected in kinematic tree: {' -> '.join(cycle)}",
            )

        # Validate joints
        for joint_name, joint in joints.items():
            self._validate_joint(joint, joint_name, result)

        # Validate transmissions
        if transmissions:
            for trans_name, trans in transmissions.items():
                self._validate_transmission(trans, trans_name, joints, result)

        return result

    def _find_reachable_links(
        self,
        root: str,
        parent_to_children: Dict[str, List[str]],
    ) -> Set[str]:
        """Find all links reachable from the root via DFS."""
        reachable = set()
        stack = [root]

        while stack:
            link = stack.pop()
            if link in reachable:
                continue
            reachable.add(link)
            stack.extend(parent_to_children.get(link, []))

        return reachable

    def _detect_cycle(
        self,
        links: List[str],
        child_to_parent: Dict[str, str],
    ) -> Optional[List[str]]:
        """
        Detect cycles in the kinematic tree.

        Returns the cycle path if found, None otherwise.
        """
        # For each link, trace back to root and check for revisits
        for start_link in links:
            visited = set()
            path = []
            current = start_link

            while current:
                if current in visited:
                    # Found cycle - extract the cycle portion
                    cycle_start = path.index(current)
                    return path[cycle_start:] + [current]

                visited.add(current)
                path.append(current)
                current = child_to_parent.get(current)

        return None

    def _validate_joint(
        self,
        joint,
        joint_name: str,
        result: ValidationResult,
    ) -> None:
        """Validate individual joint."""
        from .types import JointType

        # Check axis normalization
        import numpy as np
        axis_norm = np.linalg.norm(joint.axis)
        if abs(axis_norm - 1.0) > 1e-6:
            result.add_issue(
                ValidationSeverity.WARNING,
                f"Joint axis not normalized (norm={axis_norm:.6f})",
                "joint",
                joint_name,
            )

        # Check limits for revolute/prismatic joints
        if joint.joint_type in (JointType.REVOLUTE, JointType.PRISMATIC):
            if joint.limit is None:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    f"Joint has no limits defined",
                    "joint",
                    joint_name,
                )
            elif joint.limit.lower >= joint.limit.upper:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    f"Joint limits invalid: lower ({joint.limit.lower}) >= upper ({joint.limit.upper})",
                    "joint",
                    joint_name,
                )

        # Check mimic joint reference
        if joint.mimic is not None:
            if not joint.mimic.joint:
                result.add_issue(
                    ValidationSeverity.ERROR,
                    "Mimic joint reference is empty",
                    "joint",
                    joint_name,
                )

    def _validate_transmission(
        self,
        trans,
        trans_name: str,
        joints: Dict[str, object],
        result: ValidationResult,
    ) -> None:
        """Validate transmission."""
        if not trans.joints:
            result.add_issue(
                ValidationSeverity.WARNING,
                "Transmission has no joints",
                "transmission",
                trans_name,
            )

        if not trans.actuators:
            result.add_issue(
                ValidationSeverity.WARNING,
                "Transmission has no actuators",
                "transmission",
                trans_name,
            )

        # Check joint references
        for trans_joint in trans.joints:
            if trans_joint.name not in joints:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    f"Transmission references unknown joint '{trans_joint.name}'",
                    "transmission",
                    trans_name,
                )


def validate_urdf(urdf) -> ValidationResult:
    """
    Convenience function to validate a URDF object.

    Args:
        urdf: URDF object to validate

    Returns:
        ValidationResult
    """
    validator = URDFValidator()
    return validator.validate(
        links=urdf._links,
        joints=urdf._joints,
        transmissions=getattr(urdf, '_transmissions', None),
    )
