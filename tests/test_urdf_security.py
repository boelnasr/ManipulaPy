#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Security regression tests for URDF mesh path resolution.

A malicious URDF must not be able to escape the robot-description directory
via a relative or ``package://`` mesh filename. Explicit ``file://`` URIs and
explicit absolute paths remain honored by design.

Copyright (c) 2025 Mohamed Aboelnasr
"""


class TestRelativePathTraversal:
    """PackageResolver must refuse escaping relative references."""

    def test_resolver_refuses_escaping_relative_path(self, tmp_path):
        """A ``../`` relative mesh path escaping base_path is refused."""
        from ManipulaPy.urdf.resolver import PackageResolver

        base = tmp_path / "robot_desc"
        base.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("top secret")

        resolver = PackageResolver(base_path=base, use_ros=False)
        result = resolver.resolve("../secret.txt")

        # Refused -> returned unchanged, never the escaping absolute path.
        assert result == "../secret.txt"
        assert str(secret) not in result

    def test_resolver_allows_relative_inside_base(self, tmp_path):
        """A legitimate relative mesh inside base_path still resolves."""
        from ManipulaPy.urdf.resolver import PackageResolver

        base = tmp_path / "robot_desc"
        (base / "meshes").mkdir(parents=True)
        mesh = base / "meshes" / "link.stl"
        mesh.write_text("solid")

        resolver = PackageResolver(base_path=base, use_ros=False)
        result = resolver.resolve("meshes/link.stl")

        assert result == str(mesh)

    def test_resolver_allows_absolute_path(self, tmp_path):
        """An explicit absolute mesh path is honored by design."""
        from ManipulaPy.urdf.resolver import PackageResolver

        mesh = tmp_path / "abs.stl"
        mesh.write_text("solid")

        resolver = PackageResolver(base_path=tmp_path, use_ros=False)
        result = resolver.resolve(str(mesh))

        assert result == str(mesh)


class TestLegacyHandlerTraversal:
    """The legacy filename handler must apply the same rejection."""

    def test_legacy_handler_refuses_package_traversal(self, tmp_path):
        """``package://pkg/../..`` through the legacy handler is refused."""
        from ManipulaPy.urdf.parser import URDFParser

        base = tmp_path / "robot_desc"
        base.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("x")

        handler = URDFParser._create_filename_handler(base_path=base, mesh_dir=None)
        result = handler("package://pkg/../secret.txt")

        assert result == "package://pkg/../secret.txt"
        assert str(secret) not in result

    def test_legacy_handler_refuses_relative_traversal(self, tmp_path):
        """A ``../`` relative path through the legacy handler is refused."""
        from ManipulaPy.urdf.parser import URDFParser

        base = tmp_path / "robot_desc"
        base.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("x")

        handler = URDFParser._create_filename_handler(base_path=base, mesh_dir=None)
        result = handler("../secret.txt")

        assert result == "../secret.txt"
        assert str(secret) not in result

    def test_legacy_handler_allows_relative_inside_base(self, tmp_path):
        """A legitimate relative mesh inside base_path still resolves."""
        from ManipulaPy.urdf.parser import URDFParser

        base = tmp_path / "robot_desc"
        (base / "meshes").mkdir(parents=True)
        mesh = base / "meshes" / "l.stl"
        mesh.write_text("s")

        handler = URDFParser._create_filename_handler(base_path=base, mesh_dir=None)

        assert handler("meshes/l.stl") == str(mesh)


class TestFromXmlStringGuard:
    """URDF.from_xml_string must wire the guarded PackageResolver."""

    def test_from_xml_string_uses_package_resolver(self):
        """The default handler is a PackageResolver, not the legacy closure."""
        from ManipulaPy.urdf import URDF
        from ManipulaPy.urdf.resolver import PackageResolver

        robot = URDF.from_xml_string('<robot name="r"><link name="base"/></robot>')
        handler = robot._filename_handler

        assert isinstance(getattr(handler, "__self__", None), PackageResolver)

    def test_from_xml_string_refuses_escaping_relative_mesh(self, tmp_path):
        """A malicious relative mesh in an XML string does not resolve out."""
        from ManipulaPy.urdf import URDF

        base = tmp_path / "robot_desc"
        base.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("x")

        xml = (
            '<robot name="r">'
            '<link name="base">'
            "<visual><geometry>"
            '<mesh filename="../secret.txt"/>'
            "</geometry></visual>"
            "</link>"
            "</robot>"
        )

        robot = URDF.from_xml_string(xml, base_path=base)
        mesh = robot.link_map["base"].visuals[0].geometry

        assert mesh.filename == "../secret.txt"
        assert str(secret) not in mesh.filename
