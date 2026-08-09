# Robotics Motion Studies Design

## Summary

Extend ManipulaPy's scientific Manim curriculum beyond kinematics with ten
short, reproducible motion studies covering dynamics, singularity analysis,
path planning, and control. The studies form one progressive laboratory: the
same seven-axis Franka Panda, poses, time base, visual grammar, and deterministic
experiment data carry across all four guides.

The animations are teaching instruments rather than decorative motion. Each
study makes one testable claim, shows the physical state beside the relevant
scientific quantity, and links the displayed result to current public
ManipulaPy APIs. Existing user guides are expanded in place; this project does
not add four competing tutorial routes or enlarge the homepage gallery.

## Decisions

- Produce a comprehensive suite rather than a four-scene pilot.
- Deliver ten studies: three dynamics, two singularity, three path-planning,
  and two control studies.
- Expand the existing four user guides instead of creating new standalone
  tutorials.
- Use the bundled Franka Panda and one deterministic experiment throughout.
- Build on the existing Manim renderer, validation, GIF optimization, static
  fallback, and documentation-builder conventions.
- Keep each study under five seconds and play each GIF once.

## Goals

- Turn dense equations and static code samples into observable state changes.
- Let readers connect robot motion to matrices, forces, conditioning,
  trajectories, and feedback signals.
- Demonstrate the behavior of real ManipulaPy methods with deterministic,
  executable data.
- Establish a coherent scientific visual language across the advanced guides.
- Keep normal documentation builds independent of Manim, FFmpeg, CUDA, and a
  display server.
- Serve accessible static figures to reduced-motion, ePub, LaTeX, and print
  readers.
- Make the complete media suite reproducible, testable, and inexpensive to
  deliver.

## Non-goals

- Do not change public package APIs or numerical algorithms merely to support
  the animations.
- Do not invent performance results, simulated sensor measurements, or planner
  behavior that the current library does not produce.
- Do not turn the guides into interactive browser simulations or add WebGL or
  charting dependencies.
- Do not require a GPU for examples, tests, rendering, or documentation builds.
- Do not autoplay forever, add playback controls to GIFs, or depend on motion
  for understanding.
- Do not add these ten studies to the homepage motion gallery in this phase.
- Do not rewrite unrelated portions of the four large guides.

## Curriculum Structure

The suite follows one Panda experiment from physical model to commanded motion:

1. Dynamics explains what torque is required and what motion that torque
   produces.
2. Singularity analysis shows where the velocity map loses capability.
3. Path planning constructs smooth, feasible motion while respecting geometry.
4. Control closes the loop and measures how accurately the robot follows that
   motion.

Each guide receives a short introduction to its visual studies, two or three
figures placed beside the concepts they explain, a **What to notice** paragraph,
and a reproducibility note naming the exact API and experiment data.

## Shared Panda Experiment

Add a Manim-free source-of-truth module under `docs/examples/` for the common
experiment. It owns:

- resolution of the bundled Franka Panda URDF;
- the seven arm joints and declared limits;
- start, intermediate, goal, and near-singular joint configurations;
- a fixed gravity vector, tool wrench, time span, and sample count;
- deterministic joint and Cartesian reference trajectories;
- obstacle geometry and clearance used by the planning study;
- controller gains, initial state, disturbance definition, and integration
  interval;
- units, numerical tolerances, and named result records consumed by scenes and
  tests.

Random sampling is avoided. If a library path requires randomness, the module
uses an explicit seed and returns the seed with its results. Importing the
module performs no rendering, writes no files, and does not require Manim.

All scene values come from public ManipulaPy methods wherever practical. Small
reference calculations may validate identities or prepare display quantities,
but must not replace the library result being taught.

## Motion Studies

### Dynamics

#### 1. Configuration-dependent mass matrix

The Panda moves from the shared start pose to the intermediate pose. A
synchronized symmetric heatmap displays
`ManipulatorDynamics.mass_matrix(theta)` while a time cursor identifies the
current configuration.

The study teaches that inertia and joint coupling depend on configuration. It
labels matrix units, maintains a fixed color scale across frames, and displays
the symmetry residual as a small validation annotation. The scale must not
renormalize per frame because that would conceal relative change.

#### 2. Inverse-dynamics torque decomposition

The planned motion plays beside aligned traces for inertial, velocity-dependent,
gravity, tool-wrench, and total torque contributions. The total is computed by
`inverse_dynamics`; components use the dynamics module's public mass-matrix,
velocity-force, and gravity-force methods with the shared state.

One representative joint is emphasized while the remaining joints stay
available as restrained context. A moving vertical cursor keeps pose and traces
synchronized. Units are newton-metres, and positive/negative torque shares one
fixed axis.

#### 3. Forward/inverse dynamics round trip

The left half shows desired joint acceleration entering inverse dynamics to
produce torque. The right half sends that torque through forward dynamics and
overlays recovered acceleration against the desired signal.

The scientific claim is numerical consistency under the same model, state,
gravity, and tool wrench. The final frame reports the maximum absolute recovery
error and the tested tolerance; it does not claim exact symbolic equality.

### Singularity Analysis

#### 4. Manipulability ellipsoid collapse

The Panda approaches the shared near-singular configuration. Beside it, a
velocity manipulability ellipsoid derived from the current Jacobian contracts
along the lost-motion direction. Principal directions and singular values use
a stable scale and consistent ordering across frames.

The study explicitly distinguishes a geometric illustration from the robot's
physical link geometry. It identifies the weakest Cartesian direction and
avoids determinant-only explanations.

#### 5. Condition monitor along a trajectory

The arm traverses a deterministic path while synchronized plots show minimum
singular value and condition number from `Singularity` analysis. A visible
threshold band changes the status from well-conditioned to near-singular at the
documented criterion.

The condition-number plot uses a scale that keeps the threshold and peak
legible. Infinite values, if produced, receive an explicit capped marker rather
than being silently discarded. The final annotation relates the diagnostic to
the pose where capability is lost.

### Path Planning

#### 6. Cubic versus quintic time scaling

Two synchronized traces use `TrajectoryPlanning.joint_trajectory` with the same
Panda endpoints and duration but different supported time-scaling methods.
Position, velocity, acceleration, and jerk are plotted on aligned axes while a
small Panda view shows the shared physical motion.

Endpoint velocity and acceleration values remain visible at the final frame so
the smoothness difference is directly inspectable. Derivatives use the
planner's returned values or its public derivative calculation, with units and
sampling interval stated.

#### 7. Joint-space versus Cartesian interpolation

Two ghosted Panda instances share the same start and goal. One follows the
joint-space trajectory; the other follows `cartesian_trajectory`. Persistent
tool-tip trails reveal the different spatial paths while an endpoint marker
confirms their common task.

The legend names the interpolation domain, not merely line colors. Pose
orientation is either included consistently or explicitly held fixed; the
study must not imply full-pose interpolation if it visualizes position alone.

#### 8. Collision-avoidance correction

The scene begins with an obstructed nominal path, then shows the corrected path
from the planner's public `plan_trajectory`/collision-avoidance flow around a
declared joint-space obstacle. The nominal path remains as a faint reference,
and the minimum measured joint-space clearance updates as the corrected
trajectory develops.

The obstacle configuration, projected joint-space exclusion region, distance
metric in radians, and safety margin are named. The scene does not depict this
joint-space potential-field input as a workspace obstacle. It passes only if
the corrected path reaches the goal and its measured minimum clearance
satisfies the documented margin. If the current public planner cannot
reproducibly satisfy that contract on the Panda fixture,
implementation must stop and revise the study design rather than animate a
hand-authored detour.

### Control

#### 9. Controller tracking comparison

Open-loop, PID, and computed-torque responses follow the same reference from
the same initial state and deterministic disturbance. The Panda view follows
the active response while aligned traces retain all three responses for direct
comparison.

The study uses `ManipulatorController.pid_control` and
`computed_torque_control` with documented gains. Integration, saturation, and
disturbance assumptions are stated. The legend never describes one controller
as universally superior; it reports behavior for this experiment.

#### 10. Tracking-performance dashboard

One computed-torque run synchronizes reference and response, tracking error,
torque effort, and the controller's rise-time, overshoot, settling-time, and
steady-state-error metrics. Metric values appear only when enough of the
response has elapsed to define them.

The final dashboard states the target joint, tolerance band, units, and exact
metric definitions used by ManipulaPy. Undefined metrics are rendered as
**not reached** rather than zero.

## Visual and Motion System

- Use the established deep-navy scientific field, restrained teal active
  signal, warm amber threshold/warning signal, and red only for violations or
  unstable states.
- Reuse the Panda geometry, typography, axes, line weights, panel framing, and
  time-cursor treatment established by the kinematics studies.
- Prefer a robot/state panel on the left and a synchronized plot/diagnostic
  panel on the right. Stack panels at narrow documentation widths.
- Preserve important historical traces at reduced opacity instead of erasing
  evidence as the animation advances.
- Use fixed axis ranges and color scales unless a log scale is scientifically
  necessary and clearly labeled.
- Keep equations compact and subordinate to the displayed relationship.
- Avoid particles, decorative sweeps, faux circuitry, elastic easing, dramatic
  camera moves, and unrelated 3D rotation.
- Keep every animation at 960 by 540 pixels, under five seconds, and single
  play. The final frame must remain useful as the static PNG.

## Scene and Renderer Architecture

Create four domain modules:

- `docs/manim/dynamics_scenes.py`
- `docs/manim/singularity_scenes.py`
- `docs/manim/path_planning_scenes.py`
- `docs/manim/control_scenes.py`

Shared Manim primitives belong in a small common module only when at least two
domains use them. Candidate primitives include the Panda chain view, framed
axes, synchronized cursor, scientific legend, metric badge, and final-frame
hold. Domain-specific charts remain in their scene module.

Extend the existing renderer into one registry-driven entry point rather than
adding four unrelated render scripts. The registry records:

- scene class and deterministic output stem;
- domain and guide destination;
- GIF and PNG expectations;
- maximum duration and file size;
- static-frame selection;
- scientific validation callback where appropriate.

The normal Sphinx build consumes committed assets and never invokes Manim.
Rendering remains an explicit maintainer workflow in the isolated, pinned
Manim environment.

## Asset Delivery Contracts

Each scene produces one optimized GIF and one crisp PNG under domain-specific
directories in `docs/source/_static/tutorials/`.

- GIFs play once and contain no NETSCAPE infinite-loop extension.
- Each GIF is shorter than five seconds and no larger than 1.25 MB.
- The ten GIFs together target at most 8 MB.
- GIF and PNG dimensions are exactly 960 by 540.
- Adaptive palette generation preserves titles, equations, traces, and warning
  colors; fidelity checks compare representative frames and the final PNG.
- The final animation frame and PNG use the same scene bounds and scientific
  state.
- Temporary MP4, palette, and frame artifacts stay outside the source tree.
- Renderer `--check` validates existing assets without requiring Manim.

Missing Manim or FFmpeg produces a concise actionable error. Scientific
validation failure, missing outputs, excessive duration/size, invalid image
shape, infinite looping, or non-finite data fails the render command.

## Documentation Integration

Expand these existing guides in place:

- `docs/source/user_guide/Dynamics.rst`
- `docs/source/user_guide/Singularity_Analysis.rst`
- `docs/source/user_guide/Path_Planning.rst`
- `docs/source/user_guide/Control.rst`

Place each study immediately after its mathematical explanation and before the
first substantial executable example for that concept. Every placement has:

1. a concise figure caption naming the relationship;
2. descriptive alt text that communicates the scientific claim;
3. a **What to notice** paragraph;
4. a reproducibility note naming the public API and shared experiment;
5. a link or marked `literalinclude` excerpt for the relevant executable code.

HTML uses the existing reduced-motion-aware GIF/PNG pattern. ePub and LaTeX use
PNG only. The prose remains complete without the image, and the static final
frame contains the conclusion rather than an arbitrary intermediate state.

No new top-level tutorial route or homepage card is introduced. Existing
headings and inbound links remain stable unless a correction is necessary for
valid structure.

## Accessibility and Responsive Behavior

- Reduced-motion readers receive the PNG before an animated asset is fetched
  where supported by the existing markup pattern.
- Alt text describes the changing relationship, not colors or layout.
- Captions and nearby prose expose thresholds, units, and conclusions in text.
- Information is never encoded by color alone; line styles, labels, or markers
  distinguish comparisons.
- Panel labels remain readable at 320 CSS pixels. Mobile presentation stacks or
  scales the complete 16:9 figure without cropping axes, legends, equations, or
  the robot.
- Explicit width and height prevent layout shift.
- Light and dark themes retain the deliberate framed media surface and adequate
  caption contrast.
- Pages must have no horizontal overflow at supported mobile widths.

## Numerical and Scientific Validation

Focused tests execute the shared experiment without Manim and assert:

- mass matrices are finite, symmetric within tolerance, and have the expected
  seven-by-seven shape;
- torque components and total torque use consistent shapes and reconstruct the
  documented relation within tolerance;
- forward dynamics recovers the acceleration supplied to inverse dynamics
  within a declared numerical tolerance;
- Jacobians and singular values are finite away from the singular fixture, and
  the near-singular fixture crosses the documented threshold;
- cubic and quintic trajectories share endpoints and satisfy their documented
  endpoint conditions;
- joint and Cartesian studies share task endpoints while producing the claimed
  distinct tool paths;
- collision-corrected motion reaches the goal and meets the declared clearance;
- every controller begins from the same state and uses the same reference and
  disturbance;
- metric functions return values consistent with their documented definitions,
  including explicit handling of undefined metrics.

Assertions use tolerances justified by the numerical backend and experiment;
they do not freeze insignificant full-array digits.

## Test and Build Strategy

### Source contracts

- Test scene registry completeness and unique output names.
- Test that all ten studies consume shared experiment results rather than
  embedded arbitrary arrays.
- Test documentation references, alt text, PNG fallbacks, explicit dimensions,
  and builder branches.
- Test that HTML is the only builder allowed to reference GIFs.
- Test marker regions used by `literalinclude` and execute the included example
  functions.

### Media contracts

- Run renderer `--check` over all twenty assets.
- Validate dimensions, frame counts, durations, loop metadata, size budgets,
  palette fidelity, final-frame agreement, and readable files.
- Inspect representative frames from every study and all final PNGs at native
  resolution before committing regenerated assets.

### Documentation builds and browser QA

- Run focused documentation and affected numerical tests.
- Build strict HTML and strict ePub.
- Generate LaTeX and verify that the new studies reference only PNGs; record
  unrelated pre-existing builder warnings separately.
- In Chromium, inspect each affected route at desktop and 320/390-pixel mobile
  widths in light and dark themes.
- Verify no horizontal overflow, no clipped scientific content, stable layout,
  readable captions, and correct reduced-motion source selection.

## Expected Implementation Surface

- four domain scene modules under `docs/manim/`;
- shared scientific scene primitives under `docs/manim/` if justified by reuse;
- the existing Manim renderer and its registry/validator tests;
- one shared, Manim-free Panda experiment under `docs/examples/`;
- focused example regions or domain helpers under `docs/examples/`;
- twenty generated assets under
  `docs/source/_static/tutorials/{dynamics,singularity,path_planning,control}/`;
- the four existing user guides;
- focused documentation/media tests, most likely in
  `tests/test_docs_tutorials.py` plus domain numerical tests where appropriate;
- `docs/manim/requirements.txt` or renderer documentation only if the existing
  pinned toolchain needs a justified update.

No public package source, notebooks, deployment configuration, or unrelated
documentation routes should change unless implementation uncovers a real defect
that requires a separately approved scope change.

## Risks and Mitigations

### Collision study reproducibility

Collision avoidance has more environmental dependencies than interpolation.
Use one fixed obstacle and CPU-capable public planner path, assert clearance,
and stop for design revision if the public implementation cannot reproduce the
claimed correction.

### Dynamics/control simulation drift

Use one documented fixed-step integration scheme and short horizon. Compare
controllers only under identical conditions and avoid general performance
claims.

### Visual density

Ten studies can overcrowd already long guides. Place figures only at conceptual
hinges, limit each frame to one dominant claim, and keep extended numeric detail
in executable examples.

### Repository and page weight

Enforce per-file and aggregate GIF budgets, adaptive-palette fidelity, and PNG
compression. Do not weaken quality checks merely to meet size limits; simplify
the scene first.

### Documentation drift

Generate scene data and displayed example code from shared callable functions,
then test those functions and all `literalinclude` markers.

## Acceptance Criteria

The feature is complete when:

- all ten approved studies and twenty paired assets exist;
- every study uses the shared Panda experiment and current ManipulaPy behavior;
- all numerical invariants and scientific claims pass focused tests;
- every GIF is single-play, under five seconds, within its size budget, and
  visually faithful to its PNG;
- all four guides place the correct study beside explanatory prose, runnable
  source, alt text, and a static fallback;
- normal documentation builds require neither Manim nor FFmpeg;
- focused tests, renderer `--check`, strict HTML, and strict ePub pass;
- LaTeX output references the ten PNGs and no new GIF-related warning;
- browser QA passes desktop/mobile, light/dark, and reduced-motion checks;
- final visual and scientific reviews find no blocking issue;
- the worktree is clean apart from preserved pre-existing user files.
