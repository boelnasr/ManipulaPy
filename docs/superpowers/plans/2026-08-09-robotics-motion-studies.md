# Robotics Motion Studies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ten deterministic, scientifically validated Panda Manim studies to the existing dynamics, singularity, path-planning, and control guides.

**Architecture:** A Manim-free experiment module computes all shared Panda data through public ManipulaPy APIs. Four focused scene modules consume immutable result records, while one registry-driven renderer creates and validates paired GIF/PNG assets. Existing guides embed builder-specific media with executable source regions, and focused tests lock numerical, delivery, accessibility, and build contracts.

**Tech Stack:** Python 3, NumPy, ManipulaPy public APIs, Manim 0.20.1, Pillow, FFmpeg, pytest, Sphinx/PyData theme, reStructuredText.

## Global Constraints

- Use the bundled seven-axis Franka Panda and one deterministic experiment throughout.
- Deliver exactly ten studies: three dynamics, two singularity, three path-planning, and two control.
- Expand the four existing user guides; add no new tutorial routes or homepage cards.
- Normal documentation builds must not import Manim or invoke FFmpeg.
- Render every GIF/PNG at exactly 960 by 540 pixels.
- Every GIF must play once, last under 5,000 ms, and be smaller than 1,250,000 bytes.
- The ten new GIFs together must be smaller than 8,000,000 bytes.
- HTML may use GIFs through the existing reduced-motion pattern; ePub and LaTeX must use PNG only.
- All calculations and examples must run on CPU without CUDA.
- Preserve pre-existing untracked `.agents/`, `.superpowers/`, and `skills-lock.json` files.
- Do not change public package APIs, notebooks, deployment configuration, or unrelated documentation.

## Plan Suite and Execution Order

The tasks below are independently reviewable sub-projects sharing explicit
interfaces. Execute them in order: foundation (Tasks 1-3), dynamics (Tasks
4-5), singularity (Task 6), path planning (Tasks 7-8), control (Tasks 9-10),
then media and integration gates (Tasks 11-12).

---

### Task 1: Shared deterministic Panda experiment

**Files:**
- Create: `docs/examples/robotics_motion_studies.py`
- Create: `tests/test_docs_motion_studies.py`
- Reuse: `docs/examples/kinematics_tutorial.py`

**Interfaces:**
- Consumes: `get_robot_urdf("panda")`, `URDFToSerialManipulator`, and the first seven revolute Panda limits.
- Produces: `PandaFixture`, `load_panda_fixture()`, `STUDY_TIME`, `START`, `MID`, `GOAL`, `NEAR_SINGULAR`, `GRAVITY`, `TOOL_WRENCH`, and `assert_finite(name, value)`.

- [ ] **Step 1: Write the failing shared-fixture contract**

```python
def test_shared_panda_fixture_is_deterministic_and_cpu_only():
    studies = load_studies()
    first = studies.load_panda_fixture()
    second = studies.load_panda_fixture()
    assert first.joint_names == tuple(f"panda_joint{i}" for i in range(1, 8))
    assert first.serial is not None and first.dynamics is not None
    assert first.urdf_path == second.urdf_path
    assert len(first.joint_limits) == 7
    for pose in (studies.START, studies.MID, studies.GOAL, studies.NEAR_SINGULAR):
        assert pose.shape == (7,)
        assert np.isfinite(pose).all()
        assert all(lo <= q <= hi for q, (lo, hi) in zip(pose, first.joint_limits))
```

- [ ] **Step 2: Run the focused test and observe the missing module**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_motion_studies.py::test_shared_panda_fixture_is_deterministic_and_cpu_only -q`

Expected: FAIL because `docs/examples/robotics_motion_studies.py` does not exist.

- [ ] **Step 3: Implement the immutable shared fixture**

```python
@dataclass(frozen=True)
class PandaFixture:
    serial: SerialManipulator
    dynamics: ManipulatorDynamics
    urdf_path: Path
    joint_names: tuple[str, ...]
    joint_limits: tuple[tuple[float, float], ...]

def load_panda_fixture() -> PandaFixture:
    urdf_path = Path(get_robot_urdf("panda"))
    processor = URDFToSerialManipulator(urdf_path, backend="builtin", load_meshes=False)
    joints = tuple(j for j in processor.robot.actuated_joints
                   if j.joint_type is JointType.REVOLUTE
                   and j.name.startswith("panda_joint"))
    if len(joints) != 7 or any(j.limit is None for j in joints):
        raise RuntimeError("Panda motion studies require seven limited arm joints")
    return PandaFixture(
        processor.serial_manipulator,
        processor.dynamics,
        urdf_path,
        tuple(j.name for j in joints),
        tuple((j.limit.lower, j.limit.upper) for j in joints),
    )
```

Define fixed `float64` state vectors exactly as follows, plus
`STUDY_TIME = np.linspace(0.0, 4.0, 61)`, zero tool wrench, and
`[0, 0, -9.81]` gravity:

```python
START = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.785])
MID = np.array([0.15, -0.45, 0.1, -1.8, 0.05, 1.45, 0.65])
GOAL = np.array([-0.35, 0.25, 0.3, -1.55, -0.2, 1.25, -0.4])
NEAR_SINGULAR = np.array([
    -1.14373813, -0.2547399, 1.3507626, -0.42578919,
    1.59465886, 1.99271463, -2.05067442,
])
OBSTACLE_Q = 0.5 * (START + GOAL)
JOINT_CLEARANCE = 0.20
```

The near-singular fixture has a measured minimum space-Jacobian singular value
of approximately `7.02e-5`, below the public `1e-4` threshold; the test must
recompute rather than freeze that approximate display value.

- [ ] **Step 4: Run the fixture test and import check**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_motion_studies.py::test_shared_panda_fixture_is_deterministic_and_cpu_only -q && python3 -m py_compile docs/examples/robotics_motion_studies.py`

Expected: PASS.

- [ ] **Step 5: Commit the foundation**

```bash
git add docs/examples/robotics_motion_studies.py tests/test_docs_motion_studies.py
git commit -m "docs: add shared Panda motion experiment"
```

### Task 2: Registry-driven renderer and delivery validator

**Files:**
- Create: `docs/manim/motion_studies_registry.py`
- Create: `docs/manim/render_motion_studies.py`
- Modify: `tests/test_docs_motion_studies.py`
- Reuse behavior from: `docs/manim/render_kinematics.py`

**Interfaces:**
- Consumes: domain scene source paths and class names added in Tasks 5, 6, 8, and 10.
- Produces: `SceneSpec(key, domain, scene_source, scene_class, filename)`, `SCENES`, `select_scenes(selection)`, `validate_media_pair(gif, png)`, and CLI `main(argv=None)` with `--scene` and `--check`.

- [ ] **Step 1: Add failing registry and validator contracts**

```python
EXPECTED_KEYS = {
    "dynamics-mass", "dynamics-torque", "dynamics-roundtrip",
    "singularity-ellipsoid", "singularity-monitor",
    "planning-scaling", "planning-domains", "planning-collision",
    "control-comparison", "control-metrics",
}

def test_motion_registry_has_ten_unique_domain_assets():
    renderer = load_motion_renderer()
    assert set(renderer.SCENES) == EXPECTED_KEYS
    specs = tuple(renderer.SCENES.values())
    assert len({spec.filename for spec in specs}) == 10
    assert Counter(spec.domain for spec in specs) == {
        "dynamics": 3, "singularity": 2, "path_planning": 3, "control": 2,
    }
```

Add Pillow-generated temporary image tests mirroring the kinematics validator,
including rejection of a looping GIF, a 5,000 ms GIF, wrong dimensions, and an
oversized file budget.

- [ ] **Step 2: Run tests and observe missing registry/renderer failures**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_motion_studies.py -k 'registry or validator' -q`

Expected: FAIL because the modules are absent.

- [ ] **Step 3: Implement the registry and generalized renderer**

```python
@dataclass(frozen=True)
class SceneSpec:
    key: str
    domain: str
    scene_source: Path
    scene_class: str
    filename: str

SCENES = {spec.key: spec for spec in (
    SceneSpec("dynamics-mass", "dynamics", MANIM_DIR / "dynamics_scenes.py",
              "PandaMassMatrixEvolution", "panda_mass_matrix_evolution"),
    SceneSpec("dynamics-torque", "dynamics", MANIM_DIR / "dynamics_scenes.py",
              "PandaTorqueDecomposition", "panda_torque_decomposition"),
    SceneSpec("dynamics-roundtrip", "dynamics", MANIM_DIR / "dynamics_scenes.py",
              "PandaDynamicsRoundTrip", "panda_dynamics_round_trip"),
    SceneSpec("singularity-ellipsoid", "singularity",
              MANIM_DIR / "singularity_scenes.py",
              "PandaManipulabilityCollapse", "panda_manipulability_collapse"),
    SceneSpec("singularity-monitor", "singularity",
              MANIM_DIR / "singularity_scenes.py",
              "PandaSingularityMonitor", "panda_singularity_monitor"),
    SceneSpec("planning-scaling", "path_planning",
              MANIM_DIR / "path_planning_scenes.py",
              "PandaTimeScalingComparison", "panda_time_scaling_comparison"),
    SceneSpec("planning-domains", "path_planning",
              MANIM_DIR / "path_planning_scenes.py",
              "PandaInterpolationDomains", "panda_interpolation_domains"),
    SceneSpec("planning-collision", "path_planning",
              MANIM_DIR / "path_planning_scenes.py",
              "PandaCollisionCorrection", "panda_collision_correction"),
    SceneSpec("control-comparison", "control", MANIM_DIR / "control_scenes.py",
              "PandaControllerComparison", "panda_controller_comparison"),
    SceneSpec("control-metrics", "control", MANIM_DIR / "control_scenes.py",
              "PandaControlMetrics", "panda_control_metrics"),
)}
```

Port the proven argument-list Manim and FFmpeg pipeline from
`render_kinematics.py`. Resolve each asset as
`ASSET_ROOT / spec.domain / f"{spec.filename}{suffix}"`. Set
`MAX_GIF_BYTES = 1_250_000`, `MAX_TOTAL_GIF_BYTES = 8_000_000`, and preserve
the current dimension, duration, no-loop, palette-level, final-frame RMS,
temporary-directory, Cairo, seed-zero, and user-config-isolation checks.

- [ ] **Step 4: Run renderer unit contracts**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_motion_studies.py -k 'registry or validator or render_command' -q`

Expected: PASS without importing Manim or requiring committed assets.

- [ ] **Step 5: Commit renderer infrastructure**

```bash
git add docs/manim/motion_studies_registry.py docs/manim/render_motion_studies.py tests/test_docs_motion_studies.py
git commit -m "docs: add motion-study media registry"
```

### Task 3: Shared scientific scene primitives

**Files:**
- Create: `docs/manim/scientific_scene.py`
- Modify: `tests/test_docs_motion_studies.py`

**Interfaces:**
- Consumes: NumPy arrays and Manim 0.20.1 objects only at render time.
- Produces: `study_title(title, subtitle)`, `panel_frame(width, height)`, `panda_chain(theta)`, `time_cursor(axes, t)`, `scientific_legend(items)`, `metric_badge(label, value, unit, status)`, and shared color constants.

- [ ] **Step 1: Add source-level failing contracts**

```python
def test_shared_scene_primitives_define_accessible_visual_tokens():
    source = read(MANIM / "scientific_scene.py")
    for symbol in ("PANEL", "TEAL", "AMBER", "VIOLATION", "panda_chain",
                   "time_cursor", "metric_badge", "study_title"):
        assert symbol in source
    assert "rate_func=linear" in source
    for forbidden in ("Flash(", "Wiggle(", "there_and_back", "random"):
        assert forbidden not in source
```

- [ ] **Step 2: Run and observe the missing source failure**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_motion_studies.py::test_shared_scene_primitives_define_accessible_visual_tokens -q`

Expected: FAIL because `scientific_scene.py` is absent.

- [ ] **Step 3: Extract only genuinely shared primitives**

Port the normalized Panda chain and title/rule geometry from
`kinematics_scenes.py`, then add stable framed axes, cursor, legend, and metric
badge helpers. Functions return Manim mobjects and do not call `Scene.play`.
Use fixed typography and semantic signals; add line styles/markers so color is
never the only comparison channel.

- [ ] **Step 4: Compile and run source contracts**

Run: `python3 -m py_compile docs/manim/scientific_scene.py && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_motion_studies.py::test_shared_scene_primitives_define_accessible_visual_tokens -q`

Expected: PASS.

- [ ] **Step 5: Commit shared primitives**

```bash
git add docs/manim/scientific_scene.py tests/test_docs_motion_studies.py
git commit -m "docs: add shared scientific scene primitives"
```

### Task 4: Dynamics experiment results and executable regions

**Files:**
- Modify: `docs/examples/robotics_motion_studies.py`
- Modify: `tests/test_docs_motion_studies.py`

**Interfaces:**
- Consumes: `PandaFixture`, fixed states, `mass_matrix`, `velocity_quadratic_forces`, `gravity_forces`, `inverse_dynamics`, `forward_dynamics`, and `serial.jacobian`.
- Produces: `DynamicsResults` and `compute_dynamics_results() -> DynamicsResults` with `(61, 7, 7)` mass matrices and `(61, 7)` component/round-trip arrays.

- [ ] **Step 1: Write failing numerical invariants**

```python
def test_dynamics_study_reconstructs_torque_and_round_trip():
    result = load_studies().compute_dynamics_results()
    assert result.mass_matrices.shape == (61, 7, 7)
    assert np.isfinite(result.mass_matrices).all()
    assert np.max(np.abs(result.mass_matrices - result.mass_matrices.swapaxes(1, 2))) < 1e-8
    reconstructed = result.inertia + result.velocity + result.gravity + result.tool
    assert np.allclose(reconstructed, result.total_torque, atol=1e-8, rtol=1e-8)
    assert np.max(np.abs(result.recovered_acceleration - result.acceleration)) < 1e-8
```

- [ ] **Step 2: Run and confirm missing computation failure**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_motion_studies.py::test_dynamics_study_reconstructs_torque_and_round_trip -q`

Expected: FAIL because `compute_dynamics_results` is undefined.

- [ ] **Step 3: Implement deterministic dynamics calculation**

Define `DynamicsResults` with time, theta/dtheta/ddtheta, matrices, four torque
components, total torque, and recovered acceleration. Generate a smooth quintic
reference between shared poses. For every sample compute:

```python
M = np.asarray(dynamics.mass_matrix(q), dtype=np.float64)
c = np.asarray(dynamics.velocity_quadratic_forces(q, dq), dtype=np.float64)
g_tau = np.asarray(dynamics.gravity_forces(q, GRAVITY), dtype=np.float64)
tool_tau = np.asarray(serial.jacobian(q).T @ TOOL_WRENCH, dtype=np.float64)
inertia = M @ ddq
total = np.asarray(dynamics.inverse_dynamics(q, dq, ddq, GRAVITY, TOOL_WRENCH))
recovered = np.asarray(dynamics.forward_dynamics(q, dq, total, GRAVITY, TOOL_WRENCH))
```

Reject non-finite values and unexpected shapes with named `RuntimeError`s.
Surround the public example calls with non-nested `literalinclude` markers.

- [ ] **Step 4: Run numerical and marker tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_motion_studies.py -k 'dynamics or marker' -q`

Expected: PASS.

- [ ] **Step 5: Commit dynamics data**

```bash
git add docs/examples/robotics_motion_studies.py tests/test_docs_motion_studies.py
git commit -m "docs: compute Panda dynamics studies"
```

### Task 5: Dynamics scenes and guide integration

**Files:**
- Create: `docs/manim/dynamics_scenes.py`
- Modify: `docs/source/user_guide/Dynamics.rst`
- Modify: `tests/test_docs_motion_studies.py`

**Interfaces:**
- Consumes: `compute_dynamics_results()` and Task 3 primitives.
- Produces: `PandaMassMatrixEvolution`, `PandaTorqueDecomposition`, and `PandaDynamicsRoundTrip`.

- [ ] **Step 1: Add failing scene and RST embedding contracts**

Assert all three class names exist, call `compute_dynamics_results`, and the
guide contains three unique media stems, three PNG fallback branches, three
**What to notice** blocks, explicit `width="960" height="540"`, alt text, and
the dynamics example marker references.

- [ ] **Step 2: Run and verify missing scene/markup failures**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_motion_studies.py -k dynamics -q`

Expected: FAIL on absent classes and media references.

- [ ] **Step 3: Implement the three scenes**

Use a fixed heatmap scale across all mass-matrix frames; a shared torque axis
for positive/negative newton-metres; and desired/recovered acceleration traces
with a final maximum-error annotation. Keep the dominant joint highlighted but
retain all seven as subdued context. Every scene ends with at least a 0.35 s
final-frame hold and uses linear time progression.

- [ ] **Step 4: Embed studies at the corresponding guide concepts**

Place mass-matrix media after its configuration-dependence explanation, torque
decomposition after inverse dynamics, and round trip between inverse and
forward dynamics. Use the existing HTML/reduced-motion and builder-only PNG
pattern from `kinematics_guide.rst`; do not duplicate raw HTML in ePub.

- [ ] **Step 5: Run focused tests and strict HTML smoke build**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_motion_studies.py -k dynamics -q && make -C docs html SPHINXSTRICT=1`

Expected: PASS with no new warning.

- [ ] **Step 6: Commit dynamics presentation**

```bash
git add docs/manim/dynamics_scenes.py docs/source/user_guide/Dynamics.rst tests/test_docs_motion_studies.py
git commit -m "docs: visualize Panda dynamics"
```

### Task 6: Singularity data, scenes, and guide integration

**Files:**
- Modify: `docs/examples/robotics_motion_studies.py`
- Create: `docs/manim/singularity_scenes.py`
- Modify: `docs/source/user_guide/Singularity_Analysis.rst`
- Modify: `tests/test_docs_motion_studies.py`

**Interfaces:**
- Produces: `SingularityResults`, `compute_singularity_results()`, `PandaManipulabilityCollapse`, and `PandaSingularityMonitor`.

- [ ] **Step 1: Write failing spectrum and threshold contracts**

```python
def test_singularity_path_crosses_public_threshold():
    result = load_studies().compute_singularity_results()
    assert result.singular_values.shape == (61, 6)
    assert np.isfinite(result.singular_values).all()
    assert result.minimum_sigma[0] > result.threshold
    assert result.minimum_sigma[-1] < result.threshold
    assert np.array_equal(result.near_singular, result.minimum_sigma < result.threshold)
    assert np.all(result.ellipsoid_radii >= 0.0)
```

- [ ] **Step 2: Run and confirm missing result failure**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_motion_studies.py -k singularity -q`

Expected: FAIL because the results/scenes are absent.

- [ ] **Step 3: Compute the real Jacobian spectrum**

Interpolate from `MID` to the tested `NEAR_SINGULAR` pose. Compute the public
space Jacobian, `np.linalg.svd(J, full_matrices=False)`, minimum singular value,
condition number with explicit infinity handling, and the public
`singularity_analysis` status. Derive the velocity ellipsoid from the linear
velocity block using singular values as radii; do not call or copy the existing
Matplotlib plotting method's reciprocal-radius visualization.

- [ ] **Step 4: Implement scenes and guide placement**

Animate a stable-scale 2D projection of the three principal velocity axes and a
log-readable condition monitor with a visible `1e-4` threshold. Cap infinite
condition-number display with an explicit infinity marker. Embed both figures
after the guide's ellipsoid and trajectory-monitoring explanations with source,
alt text, units, and **What to notice** prose.

- [ ] **Step 5: Test and commit singularity studies**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_motion_studies.py -k singularity -q && make -C docs html SPHINXSTRICT=1`

Expected: PASS.

```bash
git add docs/examples/robotics_motion_studies.py docs/manim/singularity_scenes.py docs/source/user_guide/Singularity_Analysis.rst tests/test_docs_motion_studies.py
git commit -m "docs: animate Panda singularity analysis"
```

### Task 7: Path-planning data and collision feasibility gate

**Files:**
- Modify: `docs/examples/robotics_motion_studies.py`
- Modify: `tests/test_docs_motion_studies.py`

**Interfaces:**
- Produces: `PlanningResults`, `compute_planning_results()`, and explicit `CollisionStudyUnavailable`.

- [ ] **Step 1: Write failing interpolation and collision contracts**

Assert cubic/quintic arrays share endpoints, quintic endpoint velocity and
acceleration are within `1e-8`, tool paths share endpoints but differ internally,
and the public collision-corrected path reaches the goal with minimum clearance
at least the declared margin.

- [ ] **Step 2: Run and verify missing computation failure**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_motion_studies.py -k planning -q`

Expected: FAIL because `compute_planning_results` is undefined.

- [ ] **Step 3: Implement CPU trajectory calculations**

Construct `TrajectoryPlanning(serial, urdf_path, dynamics, limits,
use_cuda=False, auto_optimize=False)`. Use public `joint_trajectory` for cubic
and quintic profiles and `cartesian_trajectory` for the spatial comparison.
Use returned derivative arrays or `calculate_derivatives` with the explicit
sample interval. Convert every public boundary to host `float64` arrays.

- [ ] **Step 4: Prove or reject the collision fixture**

Call public `plan_trajectory(START, GOAL, [OBSTACLE_Q])` with the planner's
configured collision/potential-field path. Because `PotentialField` defines
obstacles in joint space, independently calculate
`min(norm(q - OBSTACLE_Q))` in radians and visualize the exclusion region in a
labeled joint-space projection; do not present it as workspace geometry. If no
fixed public configuration reaches the goal and satisfies `JOINT_CLEARANCE`,
raise `CollisionStudyUnavailable` and stop this task for a design correction;
do not synthesize a detour. Record the exact obstacle, margin, and public call in
the example region.

- [ ] **Step 5: Run deterministic planning tests twice**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_motion_studies.py -k planning -q && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_motion_studies.py -k planning -q`

Expected: both runs PASS with byte-equal numeric arrays.

- [ ] **Step 6: Commit planning data**

```bash
git add docs/examples/robotics_motion_studies.py tests/test_docs_motion_studies.py
git commit -m "docs: compute Panda planning studies"
```

### Task 8: Path-planning scenes and guide integration

**Files:**
- Create: `docs/manim/path_planning_scenes.py`
- Modify: `docs/source/user_guide/Path_Planning.rst`
- Modify: `tests/test_docs_motion_studies.py`

**Interfaces:**
- Consumes: `compute_planning_results()`.
- Produces: `PandaTimeScalingComparison`, `PandaInterpolationDomains`, and `PandaCollisionCorrection`.

- [ ] **Step 1: Add failing three-scene and guide contracts**

Assert registry class names, data consumption, persistent path traces, cubic
and quintic line-style labels, obstacle/safety-margin text, three builder-safe
media blocks, three **What to notice** sections, and no new top-level route.

- [ ] **Step 2: Run and verify red state**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_motion_studies.py -k planning -q`

Expected: FAIL on absent scenes/markup.

- [ ] **Step 3: Implement the three planning scenes**

Keep aligned position/velocity/acceleration/jerk axes for time scaling. Use two
ghosted Panda chains and persistent end-effector trails for interpolation. Show
the nominal collision path throughout correction, label obstacle geometry and
minimum clearance, and end at the common goal.

- [ ] **Step 4: Integrate and verify**

Place studies beside time scaling, Cartesian trajectory, and collision
avoidance sections. Run:

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_motion_studies.py -k planning -q && make -C docs html SPHINXSTRICT=1`

Expected: PASS.

- [ ] **Step 5: Commit path-planning presentation**

```bash
git add docs/manim/path_planning_scenes.py docs/source/user_guide/Path_Planning.rst tests/test_docs_motion_studies.py
git commit -m "docs: visualize Panda path planning"
```

### Task 9: Deterministic control simulation and metrics

**Files:**
- Modify: `docs/examples/robotics_motion_studies.py`
- Modify: `tests/test_docs_motion_studies.py`

**Interfaces:**
- Produces: `ControlRun`, `ControlResults`, `simulate_control(mode)`, and `compute_control_results()`.

- [ ] **Step 1: Write failing equal-condition and metric tests**

```python
def test_control_runs_share_conditions_and_report_undefined_metrics():
    result = load_studies().compute_control_results()
    assert set(result.runs) == {"open_loop", "pid", "computed_torque"}
    for run in result.runs.values():
        assert run.theta.shape == run.torque.shape == (61, 7)
        assert np.isfinite(run.theta).all() and np.isfinite(run.torque).all()
        assert np.array_equal(run.reference, result.reference)
    assert result.runs["computed_torque"].rms_error < result.runs["open_loop"].rms_error
    assert np.isinf(result.metrics["settling_time"]) or result.metrics["settling_time"] >= 0
```

- [ ] **Step 2: Run and verify missing simulation failure**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_motion_studies.py -k control -q`

Expected: FAIL because control results are absent.

- [ ] **Step 3: Implement fixed-step plant simulation**

Use three fresh `ManipulatorController` instances so integral state never leaks
between modes. At each shared sample, compute open-loop zero torque, public
`pid_control`, or public `computed_torque_control`; add the same deterministic
disturbance; clip only to one documented torque limit; advance the same
`forward_dynamics` plant with semi-implicit Euler:

```python
ddq = np.asarray(dynamics.forward_dynamics(q, dq, tau + disturbance, GRAVITY, ZERO_WRENCH))
dq = dq + ddq * dt
q = q + dq * dt
```

Calculate the target joint's public rise time, percent overshoot, settling time,
and steady-state error. Preserve `float("inf")` as undefined/not reached.
Choose gains by deterministic test, not by claiming universal superiority.

- [ ] **Step 4: Run simulation twice and assert deterministic equality**

Run the control-focused test twice as in Task 7. Expected: PASS both times with
finite state/torque arrays and identical results.

- [ ] **Step 5: Commit control data**

```bash
git add docs/examples/robotics_motion_studies.py tests/test_docs_motion_studies.py
git commit -m "docs: compute Panda control studies"
```

### Task 10: Control scenes and guide integration

**Files:**
- Create: `docs/manim/control_scenes.py`
- Modify: `docs/source/user_guide/Control.rst`
- Modify: `tests/test_docs_motion_studies.py`

**Interfaces:**
- Consumes: `compute_control_results()`.
- Produces: `PandaControllerComparison` and `PandaControlMetrics`.

- [ ] **Step 1: Add failing scene and guide contracts**

Assert both class names, the shared data call, three distinct controller line
styles/labels, tolerance-band markup, **not reached** support, two builder-safe
media blocks, metric definitions, units, source marker links, and **What to
notice** prose.

- [ ] **Step 2: Run and verify red state**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_motion_studies.py -k control -q`

Expected: FAIL on absent scenes/markup.

- [ ] **Step 3: Implement scenes and place them in the guide**

Retain open-loop, PID, computed-torque, and reference traces for the entire
comparison. Synchronize the active Panda state with the plot cursor. Reveal
metric badges only after each metric becomes defined; show infinity as **not
reached**, never `0`. Place comparison after computed-torque theory and the
dashboard before performance metrics examples.

- [ ] **Step 4: Test, strict-build, and commit**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_motion_studies.py -k control -q && make -C docs html SPHINXSTRICT=1`

Expected: PASS.

```bash
git add docs/manim/control_scenes.py docs/source/user_guide/Control.rst tests/test_docs_motion_studies.py
git commit -m "docs: animate Panda control studies"
```

### Task 11: Render, inspect, optimize, and commit twenty assets

**Files:**
- Create: `docs/source/_static/tutorials/dynamics/*.{gif,png}`
- Create: `docs/source/_static/tutorials/singularity/*.{gif,png}`
- Create: `docs/source/_static/tutorials/path_planning/*.{gif,png}`
- Create: `docs/source/_static/tutorials/control/*.{gif,png}`
- Modify: `tests/test_docs_motion_studies.py`

**Interfaces:**
- Consumes: complete registry and all scene modules.
- Produces: twenty committed assets accepted by `render_motion_studies.py --check`.

- [ ] **Step 1: Add the committed-media contract and observe red**

For every `SceneSpec`, assert readable GIF/PNG pairs, exact dimensions,
multi-frame GIF, no loop metadata, `< 5,000 ms`, `< 1,250,000 bytes`, aggregate
`< 8,000,000 bytes`, at least 32 levels per color channel, and final-frame RMS
within the renderer thresholds.

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_motion_studies.py -k committed_media -q`

Expected: FAIL because assets do not exist.

- [ ] **Step 2: Render through the pinned isolated environment**

Run the existing pinned Manim 0.20.1 environment, or create a disposable one
outside the repository from `docs/manim/requirements.txt`, then execute:

`python docs/manim/render_motion_studies.py --scene all`

Do not add system packages or repository runtime dependencies.

- [ ] **Step 3: Inspect native PNGs and representative GIF frames**

Create contact sheets in `/tmp` only. Inspect titles, equations, axes, units,
legends, threshold lines, robot/plot alignment, final state, palette fidelity,
and any clipping. Correct scene source and rerender the affected pair; never
patch generated pixels by hand.

- [ ] **Step 4: Run all media validators**

Run: `python docs/manim/render_motion_studies.py --check && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_motion_studies.py -k committed_media -q`

Expected: all ten GIF/PNG pairs valid and aggregate budget valid.

- [ ] **Step 5: Commit generated assets and any reviewed source correction**

```bash
git add docs/manim docs/source/_static/tutorials tests/test_docs_motion_studies.py
git commit -m "docs: add advanced robotics motion assets"
```

### Task 12: Full documentation, browser, and branch gate

**Files:**
- Modify only if a verified in-scope defect is found: four guides, scene/data/renderer modules, or focused tests.
- Record evidence in: `.superpowers/sdd/2026-08-09-robotics-motion-studies/final-report.md` (ignored report; do not force-add).

**Interfaces:**
- Consumes: all prior deliverables.
- Produces: a clean, evidence-backed branch ready to merge.

- [ ] **Step 1: Run complete focused tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  tests/test_docs_design.py tests/test_docs_tutorials.py \
  tests/test_docs_motion_studies.py tests/test_dynamics.py \
  tests/test_singularity.py tests/test_trajectory_planning.py \
  tests/test_control.py -q
```

Expected: PASS. If an ambient optional plugin fails before collection, rerun
with plugin autoload disabled as shown and record the unrelated environment
issue.

- [ ] **Step 2: Run executable and renderer checks**

Run: `python3 -m py_compile docs/examples/robotics_motion_studies.py docs/manim/*.py && python docs/manim/render_motion_studies.py --check`

Expected: PASS.

- [ ] **Step 3: Build all documentation formats**

Run:

```bash
make -C docs html SPHINXSTRICT=1
make -C docs epub SPHINXSTRICT=1
make -C docs latex SPHINXSTRICT=1
```

Expected: strict HTML/ePub pass. LaTeX must reference all ten new PNGs and no
new GIFs; isolate and record only genuinely pre-existing external badge
warnings.

- [ ] **Step 4: Inspect generated builder markup**

Search HTML for ten GIF and ten PNG sources, ePub/LaTeX for PNG-only references,
and verify each guide contains its assigned 3/2/3/2 studies exactly once. Scan
source and generated output for missing assets, duplicate IDs, broken
`literalinclude` markers, and raw GIF references outside HTML-only branches.

- [ ] **Step 5: Perform real Chromium QA**

Serve `docs/build/html` locally and inspect all four routes at 1440, 390, and
320 CSS pixels, light and dark themes. Emulate `prefers-reduced-motion: reduce`.
Verify current source selection, explicit dimensions, no horizontal overflow,
no clipped plots/labels, readable captions, consistent media frames, and stable
layout. Capture evidence under `/tmp`.

- [ ] **Step 6: Run final diff and scientific review**

Run `git diff --check`, inspect `git diff --stat` and every changed path, verify
the ten scene claims against their result arrays, and confirm no public package,
notebook, homepage, or deployment file changed. Fix only proven in-scope
defects, rerun the affected test plus the full focused gate, and append evidence
to the ignored report.

- [ ] **Step 7: Commit final verified corrections**

```bash
git add docs tests
git commit -m "fix: complete robotics motion study QA"
```

Skip the commit when Task 12 requires no tracked correction. End with
`git status --short`, preserving only the three pre-existing untracked user
paths named in Global Constraints.
