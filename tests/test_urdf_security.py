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
        (ws / "mypkg" / "package.xml").write_text(
            "<package><name>mypkg</name></package>"
        )
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


class TestDeepPackageDiscovery:
    """Bounded workspace discovery supports deeply nested URDF layouts."""

    def test_discovers_package_from_deeply_nested_urdf(self, tmp_path):
        """A marked workspace may contain a package beyond the old ancestor scan."""
        from ManipulaPy.urdf.resolver import PackageResolver

        workspace = tmp_path / "workspace"
        (workspace / ".git").mkdir(parents=True)
        (workspace / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        package = workspace / "src" / "vendor" / "arm_description"
        mesh = package / "meshes" / "arm.stl"
        mesh.parent.mkdir(parents=True)
        mesh.write_text("solid arm")
        (package / "package.xml").write_text(
            "<package><name>arm_description</name></package>"
        )
        urdf_dir = workspace / "inputs" / "robots" / "arm" / "models" / "urdf"
        urdf_dir.mkdir(parents=True)

        resolver = PackageResolver(base_path=urdf_dir, use_ros=False)

        assert resolver.resolve("package://arm_description/meshes/arm.stl") == str(mesh)

    def test_rejects_invalid_package_name_before_ros_lookup(
        self, tmp_path, monkeypatch
    ):
        """Option-like package names never reach ROS or catkin discovery."""
        from ManipulaPy.urdf.resolver import PackageResolver

        resolver = PackageResolver(base_path=tmp_path, use_ros=True)

        def fail_if_called(package_name):
            raise AssertionError(f"unsafe package name reached ROS: {package_name}")

        monkeypatch.setattr(resolver, "_find_ros_package", fail_if_called)

        assert resolver.resolve("package://--version/meshes/arm.stl") == (
            "package://--version/meshes/arm.stl"
        )

    def test_incomplete_git_directory_is_not_a_workspace_boundary(self, tmp_path):
        """An arbitrary empty ``.git`` directory cannot authorize a tree scan."""
        from ManipulaPy.urdf.resolver import PackageResolver

        boundary = tmp_path / "boundary"
        (boundary / ".git").mkdir(parents=True)
        package = boundary / "src" / "arm_description"
        mesh = package / "meshes" / "arm.stl"
        mesh.parent.mkdir(parents=True)
        mesh.write_text("private")
        (package / "package.xml").write_text(
            "<package><name>arm_description</name></package>"
        )
        urdf_dir = boundary / "inputs" / "deep"
        urdf_dir.mkdir(parents=True)
        resolver = PackageResolver(base_path=urdf_dir, use_ros=False)
        uri = "package://arm_description/meshes/arm.stl"

        assert resolver.resolve(uri) == uri

    def test_discovers_nested_package_under_explicit_search_root(self, tmp_path):
        """A caller-permitted search root is also a bounded discovery root."""
        from ManipulaPy.urdf.resolver import PackageResolver

        search_root = tmp_path / "packages"
        package = search_root / "vendor" / "arm_description"
        mesh = package / "meshes" / "arm.stl"
        mesh.parent.mkdir(parents=True)
        mesh.write_text("solid arm")
        (package / "package.xml").write_text(
            "<package><name>arm_description</name></package>"
        )
        resolver = PackageResolver(search_paths=[search_root], use_ros=False)

        assert resolver.resolve("package://arm_description/meshes/arm.stl") == str(mesh)

    def test_ambient_ros_path_is_direct_only(self, tmp_path, monkeypatch):
        """Ambient ROS roots are not recursively indexed for nested packages."""
        from ManipulaPy.urdf.resolver import PackageResolver

        ros_root = tmp_path / "ros_packages"
        package = ros_root / "vendor" / "arm_description"
        mesh = package / "meshes" / "arm.stl"
        mesh.parent.mkdir(parents=True)
        mesh.write_text("private")
        (package / "package.xml").write_text(
            "<package><name>arm_description</name></package>"
        )
        monkeypatch.setenv("ROS_PACKAGE_PATH", str(ros_root))
        monkeypatch.delenv("AMENT_PREFIX_PATH", raising=False)
        monkeypatch.delenv("MANIPULAPY_PACKAGE_PATH", raising=False)
        resolver = PackageResolver(use_ros=True)
        monkeypatch.setattr(resolver, "_find_ros_package", lambda _name: None)
        uri = "package://arm_description/meshes/arm.stl"

        assert resolver.resolve(uri) == uri

    def test_custom_manipulapy_path_remains_a_deep_root(self, tmp_path, monkeypatch):
        """The explicit custom path retains bounded nested-package discovery."""
        from ManipulaPy.urdf.resolver import PackageResolver

        custom_root = tmp_path / "custom_packages"
        package = custom_root / "vendor" / "arm_description"
        mesh = package / "meshes" / "arm.stl"
        mesh.parent.mkdir(parents=True)
        mesh.write_text("solid arm")
        (package / "package.xml").write_text(
            "<package><name>arm_description</name></package>"
        )
        monkeypatch.setenv("MANIPULAPY_PACKAGE_PATH", str(custom_root))
        monkeypatch.delenv("ROS_PACKAGE_PATH", raising=False)
        monkeypatch.delenv("AMENT_PREFIX_PATH", raising=False)
        resolver = PackageResolver(use_ros=False)

        assert resolver.resolve("package://arm_description/meshes/arm.stl") == str(mesh)

    def test_search_root_refuses_symlinked_direct_package_escape(self, tmp_path):
        """A direct package symlink cannot widen its caller-permitted root."""
        from ManipulaPy.urdf.resolver import PackageResolver

        search_root = tmp_path / "packages"
        search_root.mkdir()
        outside_package = tmp_path / "outside" / "arm_description"
        mesh = outside_package / "meshes" / "arm.stl"
        mesh.parent.mkdir(parents=True)
        mesh.write_text("private")
        (search_root / "arm_description").symlink_to(
            outside_package, target_is_directory=True
        )
        resolver = PackageResolver(search_paths=[search_root], use_ros=False)
        uri = "package://arm_description/meshes/arm.stl"

        assert resolver.resolve(uri) == uri

    def test_non_package_xml_cannot_authorize_upward_discovery(self, tmp_path):
        """Only a ROS ``<package>`` document may define a package boundary."""
        from ManipulaPy.urdf.resolver import PackageResolver

        boundary = tmp_path / "workspace"
        mesh = boundary / "meshes" / "arm.stl"
        mesh.parent.mkdir(parents=True)
        mesh.write_text("private")
        (boundary / "package.xml").write_text(
            "<metadata><name>arm_description</name></metadata>"
        )
        urdf_dir = boundary / "inputs" / "deep"
        urdf_dir.mkdir(parents=True)
        resolver = PackageResolver(base_path=urdf_dir, use_ros=False)
        uri = "package://arm_description/meshes/arm.stl"

        assert resolver.resolve(uri) == uri

    def test_candidate_cap_exhaustion_discards_all_discovery_results(
        self, tmp_path, monkeypatch
    ):
        """An incomplete package index cannot be combined with another root."""
        import ManipulaPy.urdf.resolver as resolver_module
        from ManipulaPy.urdf.resolver import PackageResolver

        incomplete_roots = []
        for name in ("a_decoy", "b_decoy"):
            package = tmp_path / name
            package.mkdir(parents=True)
            (package / "package.xml").write_text(
                f"<package><name>{name}</name></package>"
            )
            incomplete_roots.append(package)

        complete = tmp_path / "complete"
        package = complete / "arm_description"
        mesh = package / "meshes" / "arm.stl"
        mesh.parent.mkdir(parents=True)
        mesh.write_text("solid arm")
        (package / "package.xml").write_text(
            "<package><name>arm_description</name></package>"
        )
        monkeypatch.setattr(resolver_module, "_MAX_DISCOVERY_CANDIDATES", 1)
        resolver = PackageResolver(
            search_paths=[*incomplete_roots, complete], use_ros=False
        )
        uri = "package://arm_description/meshes/arm.stl"

        assert resolver.resolve(uri) == uri

    def test_upward_discovery_stops_at_depth_cap(self, tmp_path, monkeypatch):
        """A workspace marker beyond the upward cap is never reached."""
        import ManipulaPy.urdf.resolver as resolver_module
        from ManipulaPy.urdf.resolver import PackageResolver

        workspace = tmp_path / "workspace"
        (workspace / ".git").mkdir(parents=True)
        (workspace / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        package = workspace / "src" / "arm_description"
        mesh = package / "meshes" / "arm.stl"
        mesh.parent.mkdir(parents=True)
        mesh.write_text("solid arm")
        (package / "package.xml").write_text(
            "<package><name>arm_description</name></package>"
        )
        urdf_dir = workspace / "a" / "b" / "c" / "d"
        urdf_dir.mkdir(parents=True)
        monkeypatch.setattr(resolver_module, "_MAX_DISCOVERY_UPWARD_DEPTH", 2)
        resolver = PackageResolver(base_path=urdf_dir, use_ros=False)
        uri = "package://arm_description/meshes/arm.stl"

        assert resolver.resolve(uri) == uri

    def test_downward_discovery_stops_at_depth_cap(self, tmp_path, monkeypatch):
        """Package manifests deeper than the downward cap are not indexed."""
        import ManipulaPy.urdf.resolver as resolver_module
        from ManipulaPy.urdf.resolver import PackageResolver

        workspace = tmp_path / "workspace"
        (workspace / ".git").mkdir(parents=True)
        (workspace / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        package = workspace / "one" / "two" / "arm_description"
        mesh = package / "meshes" / "arm.stl"
        mesh.parent.mkdir(parents=True)
        mesh.write_text("solid arm")
        (package / "package.xml").write_text(
            "<package><name>arm_description</name></package>"
        )
        urdf_dir = workspace / "inputs"
        urdf_dir.mkdir()
        monkeypatch.setattr(resolver_module, "_MAX_DISCOVERY_DOWNWARD_DEPTH", 1)
        resolver = PackageResolver(base_path=urdf_dir, use_ros=False)
        uri = "package://arm_description/meshes/arm.stl"

        assert resolver.resolve(uri) == uri

    def test_discovery_refuses_symlinked_package_escape(self, tmp_path):
        """A package-directory symlink cannot escape a discovered workspace."""
        from ManipulaPy.urdf.resolver import PackageResolver

        workspace = tmp_path / "workspace"
        (workspace / ".git").mkdir(parents=True)
        (workspace / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        outside_package = tmp_path / "outside" / "arm_description"
        mesh = outside_package / "meshes" / "arm.stl"
        mesh.parent.mkdir(parents=True)
        mesh.write_text("private")
        (outside_package / "package.xml").write_text(
            "<package><name>arm_description</name></package>"
        )
        (workspace / "src").mkdir()
        (workspace / "src" / "arm_description").symlink_to(
            outside_package, target_is_directory=True
        )
        urdf_dir = workspace / "inputs" / "deep"
        urdf_dir.mkdir(parents=True)
        resolver = PackageResolver(base_path=urdf_dir, use_ros=False)
        uri = "package://arm_description/meshes/arm.stl"

        assert resolver.resolve(uri) == uri

    def test_deep_discovery_refuses_ambiguous_packages(self, tmp_path):
        """Two distinct validated package roots remain fail-closed."""
        from ManipulaPy.urdf.resolver import PackageResolver

        workspace = tmp_path / "workspace"
        (workspace / ".git").mkdir(parents=True)
        (workspace / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        for vendor in ("one", "two"):
            package = workspace / "src" / vendor / "arm_description"
            mesh = package / "meshes" / "arm.stl"
            mesh.parent.mkdir(parents=True)
            mesh.write_text(f"solid {vendor}")
            (package / "package.xml").write_text(
                "<package><name>arm_description</name></package>"
            )
        urdf_dir = workspace / "inputs" / "deep"
        urdf_dir.mkdir(parents=True)
        resolver = PackageResolver(base_path=urdf_dir, use_ros=False)
        uri = "package://arm_description/meshes/arm.stl"

        assert resolver.resolve(uri) == uri

    def test_canonical_duplicate_candidates_are_not_ambiguous(self, tmp_path):
        """Aliases to one permitted package deduplicate by canonical target."""
        from pathlib import Path

        from ManipulaPy.urdf.resolver import PackageResolver

        packages = tmp_path / "packages"
        package = packages / "arm_description"
        mesh = package / "meshes" / "arm.stl"
        mesh.parent.mkdir(parents=True)
        mesh.write_text("solid arm")
        alias = tmp_path / "arm_alias"
        alias.symlink_to(package, target_is_directory=True)
        resolver = PackageResolver(search_paths=[packages, alias], use_ros=False)

        result = resolver.resolve("package://arm_description/meshes/arm.stl")

        assert result != "package://arm_description/meshes/arm.stl"
        assert Path(result).resolve() == mesh.resolve()
