ManipulaPy.urdf — URDF Subpackage (v1.3.2+)
============================================

.. versionadded:: 1.3.2
   New native NumPy 2.0-compatible URDF parser introduced as the
   recommended alternative to the legacy ``urdf_processor`` module.

Module overview
---------------

``ManipulaPy.urdf`` is a self-contained, NumPy 2.0-compatible URDF
parser written specifically for ManipulaPy. It has zero hard
dependencies on third-party URDF libraries (``trimesh`` and
``pybullet`` are loaded lazily and only when meshes or alternative
backends are needed). The parser understands ``package://`` URIs,
``file://`` URIs, ROS 1 / ROS 2 package discovery, and Xacro macro
expansion. Loading a URDF returns a :class:`~ManipulaPy.urdf.URDF`
object that can be converted directly into a
:class:`~ManipulaPy.kinematics.SerialManipulator` or a
:class:`~ManipulaPy.dynamics.ManipulatorDynamics` instance.

URDF class
----------

.. autoclass:: ManipulaPy.urdf.URDF
   :members:
   :undoc-members:
   :show-inheritance:

Use :meth:`URDF.load` to parse a URDF (or Xacro) file from disk. The
``backend`` argument selects the parser implementation:

- ``"builtin"`` *(default)* — the native ManipulaPy parser. NumPy 2.0
  compatible, no extra installs required.
- ``"urchin"`` — legacy ``urchin`` backend. Requires the ``[urdf]``
  extra (``pip install ManipulaPy[urdf]``). Not compatible with NumPy
  2.0 and emits a ``DeprecationWarning``.
- ``"pybullet"`` — PyBullet-based loader. Requires the ``[simulation]``
  extra (``pip install ManipulaPy[simulation]``).

Once loaded, call :meth:`URDF.to_serial_manipulator` to obtain a
kinematics-ready :class:`SerialManipulator`, or
:meth:`URDF.to_manipulator_dynamics` to obtain a fully populated
:class:`ManipulatorDynamics` (mass, inertia, screw axes, joint
limits)::

   from ManipulaPy.urdf import URDF
   from ManipulaPy.ManipulaPy_data import get_robot_urdf

   robot = URDF.load(get_robot_urdf("ur5"), backend="builtin")
   serial = robot.to_serial_manipulator()
   dynamics = robot.to_manipulator_dynamics()

PackageResolver class
---------------------

.. autoclass:: ManipulaPy.urdf.PackageResolver
   :members:
   :undoc-members:

Mesh path resolution
~~~~~~~~~~~~~~~~~~~~

When a URDF references meshes via ``package://<name>/<path>``,
``PackageResolver`` walks the following strategies, in order, to map
the URI to a real file on disk:

1. **Explicit** :meth:`PackageResolver.add_package` **mapping** —
   highest precedence. Once a package is pinned with
   ``add_package(name, path)``, the resolver never falls through to
   the other strategies for that package: if the requested file does
   not exist under the pinned root, the original URI is returned with
   a warning. This is the documented escape hatch for ambiguous
   workspaces.
2. **Search paths** — every entry in ``search_paths`` is tried in
   both *package-rooted* (``<search_path>/<package_name>/<rel>``) and
   *flat* (``<search_path>/<rel>``) form.
3. **ROS package discovery** — only when ``use_ros=True`` (the
   default). Uses ``ament_index_python`` (ROS 2), then ``rospkg`` /
   ``catkin_find`` (ROS 1), then any directories listed in
   ``ROS_PACKAGE_PATH`` and ``AMENT_PREFIX_PATH``.
4. **base_path fallback** — ``<base_path>/<rel>``, useful when the
   URDF lives next to its meshes.
5. **Ancestor heuristic** — checks up to three parent directory
   levels above ``base_path`` for either ``<ancestor>/<package>/<rel>``
   or ``<ancestor>/<rel>``. This catches the common
   ``package/urdf/robot.urdf`` and ``package/robots/model.urdf``
   layouts.

Candidates from strategies 2-5 are deduplicated by their canonical
(symlink-resolved) path before the ambiguity check, so symlinked or
case-insensitive workspaces do not cause spurious "multiple matches"
warnings. If two genuinely distinct files match a single
``package://`` URI, the resolver refuses to guess and emits a warning
recommending an explicit :meth:`add_package` call to disambiguate.

Two security-conscious behaviours are worth noting:

- ``..`` traversal segments inside ``package://name/...`` paths are
  rejected with a warning — the resolver returns the original URI
  rather than escaping the package root.
- ``file://`` URIs are parsed via :func:`urllib.parse.urlparse` and
  :func:`urllib.request.url2pathname`, so Windows-style
  ``file:///C:/...`` URIs resolve correctly.

Examples
--------

.. code-block:: python

   from ManipulaPy.urdf import URDF, PackageResolver
   from ManipulaPy.ManipulaPy_data import get_robot_urdf

   # Default load with native parser (NumPy 2.0 compatible)
   robot = URDF.load(get_robot_urdf("ur5"))
   serial = robot.to_serial_manipulator()
   dynamics = robot.to_manipulator_dynamics()

   # Custom resolver for unbundled meshes — pass package_map through
   # the parser for the same effect as resolver.add_package(...).
   from ManipulaPy.urdf.parser import URDFParser
   robot = URDFParser.parse_file(
       "custom.urdf",
       package_map={"ur_description": "/opt/ros/jazzy/share/ur_description"},
       load_meshes=True,
   )

   # Or build a resolver directly for advanced use:
   resolver = PackageResolver()
   resolver.add_package("ur_description", "/opt/ros/jazzy/share/ur_description")
   resolved = resolver.resolve("package://ur_description/meshes/ur5/visual/base.dae")

   # Force the legacy urchin backend (requires `pip install ManipulaPy[urdf]`)
   robot = URDF.load("custom.urdf", backend="urchin")

See also
--------

- :doc:`urdf_processor` — legacy ``URDFToSerialManipulator`` API (still
  supported, but the native parser is recommended for new code).
