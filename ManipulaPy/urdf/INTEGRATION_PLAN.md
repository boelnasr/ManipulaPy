# URDF Module Integration Plan

Status summary:
- `parser.py` handles links/joints/materials/visuals/collisions/inertials/origins; parses limits, mimic, safety, calibration, dynamics; geometry (box/cylinder/sphere/mesh) with filename resolution; xacro supported.
- `core.py` builds a single kinematic chain, FK (with mimic, batch FK), screw axis extraction for revolute/prismatic, simple inertia defaults, conversion to SerialManipulator/ManipulatorDynamics.
- Gaps: no transmissions parsed; no scene/multi-root support (multiple roots rejected); planar/floating joints not handled in kinematics/dynamics; minimal path resolution; no graph validation beyond single-root; collision/visual data unused downstream; URDFToSerialManipulator still uses urchin; no backend selection; no scene/robot manager.

Plan:
1) Parsing completeness
   - Add transmission parsing/storage.
   - Enhance path resolution: configurable search paths/package map for `package://` and relative paths; better mesh dir handling.
   - Graph validation: detect cycles, multiple roots; either support multiple roots or expose a scene layer to split into submodels.
2) Kinematics/dynamics support
   - Handle planar/floating joints explicitly (or warn if unsupported for SerialManipulator).
   - Improve inertia handling: include COM offsets in spatial inertia; warn on missing inertials with sensible defaults.
   - Allow tip selection for home pose `M`; handle branches (primary end link or per-tip M).
3) Scene/multi-robot
   - Add a Scene class to load multiple URDFs with base transforms and namespacing to avoid link/joint clashes.
   - Support parsing unconnected subgraphs as multiple models.
4) Integration
   - Update `URDFToSerialManipulator` to use the new parser as `backend="builtin"`; optional fallback to urchin/pybullet.
   - Expose collision/visual geometry for simulators/collision checking.
5) Testing
   - Fixtures for mimic, transmissions, mesh vs. primitives, planar/floating, multi-root files.
   - Cross-check parsed kinematics/limits/inertials against pybullet/urdfdom outputs for the same URDFs.
6) Docs
   - Document parser features/limits, backend selection, path resolution, multi-robot/scene handling, and tip selection.
