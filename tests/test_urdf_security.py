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


class TestPackageUriContainment:
    """``package://`` resolution must not escape the permitted roots.

    The traversal guard in ``_resolve_package_uri`` only rejects a *literal*
    ``..`` component. The ancestor heuristic walks up from ``base_path`` on the
    resolver's own initiative, so a malicious URI needs no ``..`` at all to
    land outside the robot-description directory.
    """

    def test_package_uri_refuses_ancestor_escape_without_dotdot(self, tmp_path):
        """An unmatched package name must not resolve to an ancestor's file.

        ``package://<name>/<path>`` where ``<name>`` matches no directory and
        no package must not fall back to joining ``<path>`` onto an ancestor
        of base_path -- that reaches files the description directory does not
        own, with no ``..`` anywhere in the URI.
        """
        from ManipulaPy.urdf.resolver import PackageResolver

        base = tmp_path / "robots" / "myrobot" / "urdf"
        base.mkdir(parents=True)
        loot = tmp_path / "robots" / "loot.stl"
        loot.write_text("private")

        resolver = PackageResolver(base_path=base, use_ros=False)
        uri = "package://totally_fake_package/loot.stl"
        result = resolver.resolve(uri)

        assert result == uri, f"escaped base_path: resolved to {result!r}"
        assert str(loot) not in result

    def test_package_uri_refuses_symlink_escape(self, tmp_path):
        """A symlink inside the tree must not launder an out-of-tree target."""
        from ManipulaPy.urdf.resolver import PackageResolver

        pkg = tmp_path / "ws" / "mypkg"
        (pkg / "meshes").mkdir(parents=True)
        outside = tmp_path / "outside.stl"
        outside.write_text("private")
        link = pkg / "meshes" / "link.stl"
        link.symlink_to(outside)

        base = pkg / "urdf"
        base.mkdir()
        resolver = PackageResolver(base_path=base, use_ros=False)
        uri = "package://mypkg/meshes/link.stl"
        result = resolver.resolve(uri)

        assert result == uri, f"followed symlink out of tree: {result!r}"

    def test_package_uri_still_resolves_standard_ros_layout(self, tmp_path):
        """Containment must not break the ordinary ROS package layout.

        ``ws/src/mypkg/{urdf,meshes}`` with the URDF in ``urdf/`` and the mesh
        in the sibling ``meshes/`` is the layout the ancestor heuristic exists
        to serve. It must keep working.
        """
        from ManipulaPy.urdf.resolver import PackageResolver

        pkg = tmp_path / "ws" / "src" / "mypkg"
        (pkg / "meshes").mkdir(parents=True)
        mesh = pkg / "meshes" / "link.stl"
        mesh.write_text("solid")
        base = pkg / "urdf"
        base.mkdir()

        resolver = PackageResolver(base_path=base, use_ros=False)
        result = resolver.resolve("package://mypkg/meshes/link.stl")

        assert result == str(mesh), f"legitimate layout broke: {result!r}"

    def test_package_uri_still_resolves_package_named_subdir(self, tmp_path):
        """``<ancestor>/<package_name>/<path>`` must keep resolving."""
        from ManipulaPy.urdf.resolver import PackageResolver

        ws = tmp_path / "ws"
        mesh = ws / "mypkg" / "meshes" / "link.stl"
        mesh.parent.mkdir(parents=True)
        mesh.write_text("solid")
        (ws / "mypkg" / "package.xml").write_text("<package><name>mypkg</name></package>")
        base = ws / "desc"
        base.mkdir()

        resolver = PackageResolver(base_path=base, use_ros=False)
        result = resolver.resolve("package://mypkg/meshes/link.stl")

        assert result == str(mesh), f"legitimate layout broke: {result!r}"

    def test_package_uri_refuses_sibling_dir_that_is_not_a_package(self, tmp_path):
        """A name-matched SIBLING directory is not a package root by itself.

        ``~/robots/urdf/robot.urdf`` is an ordinary layout, which puts the home
        directory two levels up. Without a package.xml requirement,
        ``package://.ssh/id_rsa`` reads the private key.
        """
        from ManipulaPy.urdf.resolver import PackageResolver

        home = tmp_path / "home"
        base = home / "robots" / "urdf"
        base.mkdir(parents=True)
        (home / ".ssh").mkdir()
        (home / ".ssh" / "id_rsa").write_text("PRIVATE KEY")

        resolver = PackageResolver(base_path=base, use_ros=False)
        uri = "package://.ssh/id_rsa"
        result = resolver.resolve(uri)

        assert result == uri, f"read outside the description tree: {result!r}"
