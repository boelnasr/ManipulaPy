# Kinematics Tutorial and Motion Gallery Design

## Summary

Replace the two inconsistent kinematics tutorials with one canonical, executable learning path built around ManipulaPy's bundled Franka Panda model. Preserve the existing `tutorials/Kinematics` route as a compatibility page, expand `tutorials/kinematics_guide` into a focused 20-minute tutorial, and support the explanations with deterministic Manim-generated scientific animations and still images.

As a companion homepage correction, recompose “See the math move” as the approved **Wide lead, calibrated pair** layout. The change fixes the current arbitrary vertical offsets and mismatched crops without changing the section's three subjects.

## Goals

- Give a first-time user one obvious and technically correct path through forward, velocity, and inverse kinematics.
- Make every displayed Python example executable against the current public API.
- Use one real robot, the bundled seven-axis Franka Panda, from setup through validation.
- Explain mathematical state changes with restrained scientific animation rather than decorative motion.
- Keep generated media available to HTML, reduced-motion, ePub, and LaTeX/PDF readers.
- Remove duplicate tutorial navigation while preserving inbound links to the legacy route.
- Correct the homepage motion-gallery hierarchy, alignment, and scientific-image treatment.

## Non-goals

- Do not change ManipulaPy's Python API, solver behavior, URDF parser, or packaged robot data.
- Do not turn the tutorial into a complete screw-theory textbook or duplicate the notebook course.
- Do not add browser-side simulation, interactive WebGL, or a JavaScript charting dependency.
- Do not require Manim to build or read the normal documentation.
- Do not replace the homepage's existing workspace, joint-trajectory, or pick-motion subjects.
- Do not fix unrelated tutorial naming, notebook numbering, or dynamics content in this scope.

## Information Architecture

### Canonical route

`docs/source/tutorials/kinematics_guide.rst` is the canonical tutorial and remains listed in `docs/source/tutorials/index.rst`.

The page is titled **Kinematics with the Franka Panda** and begins with a concise outcome statement: in about 20 minutes the reader will load the robot, compute a pose, map joint rates to tool velocity, solve a reachable inverse-kinematics target, and verify the result numerically.

### Compatibility route

`docs/source/tutorials/Kinematics.rst` remains buildable at its current URL but stops duplicating examples. It contains:

- a clear “This tutorial has moved” notice;
- a direct link to `kinematics_guide`;
- a short explanation that the old URL is retained for bookmarks and external links;
- no separate tutorial content and no second entry in the tutorials toctree.

This preserves the route while making the canonical page unambiguous.

### Relationship to deeper material

The canonical tutorial ends with next-step links rather than expanding indefinitely:

- the notebook course for longer executable investigations;
- the conceptual Kinematics user guide for theory;
- the Kinematics API reference for signatures and parameters;
- the URDF processor guide for alternate robots;
- the singularity material for deeper conditioning analysis.

## Tutorial Narrative

The page follows one continuous Franka Panda task. It does not introduce disconnected toy arms or switch dimensions between examples.

### 1. Before you begin

- State Python 3.9 or newer, matching package metadata.
- Link to installation rather than repeating the full installation guide.
- Require only the normal ManipulaPy scientific stack for tutorial execution.
- Explain that the bundled URDF is resolved from the installed `ManipulaPy` package, not from a repository-relative `resources/` path.
- Give expected runtime and result shapes so users can recognize a successful setup.

### 2. Load one real robot

- Resolve `ManipulaPy/ManipulaPy_data/franka_panda/panda.urdf` through package resources.
- Construct `URDFToSerialManipulator` with the built-in parser and obtain its `serial_manipulator`.
- Define the tutorial's arm view as the first seven revolute Panda joints and
  derive their names from the loaded URDF rather than hard-coding six axes.
- Explain the bundled-model wrinkle once: the public processor also exposes one
  actuated gripper joint, while this arm-kinematics path consistently passes a
  seven-element arm vector. Display the seven arm names or limits as an early
  verification checkpoint.

### 3. Forward kinematics

- Define one valid seven-joint seed configuration within the Panda limits.
- Call `forward_kinematics(theta, frame="space")`.
- Explain the rotation and translation blocks of the returned 4 by 4 transform.
- Validate the homogeneous last row and rotation orthonormality with executable assertions.
- Show the first Manim study immediately after the result, so the transform and physical chain remain adjacent.

### 4. Jacobian and velocity mapping

- Compute `J = robot.jacobian(theta, frame="space")` and state its 6 by 7 shape.
- Map a small seven-element joint-rate vector through `end_effector_velocity`.
- Separate angular and linear components explicitly; do not imply that all six values share one physical unit.
- Report rank or singular values as descriptive diagnostics, without promising a warning that the library does not emit.
- Use the second Manim study to connect changing joint rates, the Jacobian, and the resulting spatial twist.

### 5. Inverse kinematics

- Generate a guaranteed-reachable target by applying forward kinematics to a second valid Panda configuration.
- Call `iterative_inverse_kinematics(T_target, theta0, ...)` using the real return order: `(theta_solution, success, iterations)`.
- Avoid nonexistent solver names such as `lm_inverse_kinematics` and `position_inverse_kinematics`.
- Explain that a redundant arm can have multiple joint-space solutions for a pose.
- Verify the returned solution by recomputing forward kinematics and measuring translation and rotation residuals instead of comparing joint vectors directly.
- Use the third Manim study for convergence, with residuals plotted on scientific axes and the target state visually fixed.

### 6. Validation and next steps

End with a compact checklist:

- pose shape is 4 by 4;
- Jacobian shape is 6 by 7;
- IK reports success for the documented target;
- the reconstructed pose is within the documented numerical tolerance;
- joint values remain within declared Panda limits.

Troubleshooting covers only observed, actionable failure modes: incorrect URDF resolution, joint-vector length, unreachable targets, poor initial guesses, and tolerance/convergence tradeoffs.

## Executable Example Architecture

Add a small source-of-truth example module under `docs/examples/kinematics_tutorial.py`.

- The module exposes named functions for locating/loading the Panda, defining deterministic configurations, computing FK/velocity/IK results, and calculating pose residuals.
- The RST page uses marked `literalinclude` regions from this module so displayed code cannot drift from tested code.
- The module has no Manim dependency and runs in a normal documentation/test environment.
- Random inputs are avoided. If sampling is ever required, it uses an explicit fixed seed.
- Importing the module performs no rendering and writes no files.
- A direct `main()` path may print the tutorial checkpoints, but computations remain callable from tests.

Tests execute the same functions used by the documentation. They check public behavior and tolerances, not a frozen full matrix whose insignificant digits may vary across supported numerical backends.

## Manim Scientific Visuals

### Asset set

Create three short studies under `docs/manim/kinematics_scenes.py`:

1. **Panda forward-kinematics chain**: a restrained joint-chain diagram builds from base to tool while the corresponding transform progression is labeled.
2. **Jacobian velocity mapping**: joint-rate inputs flow through a Jacobian matrix into angular and linear tool-velocity components, with units kept distinct.
3. **Inverse-kinematics convergence**: translation and rotation residuals decrease over iterations toward a fixed reachable target.

The scenes visualize data produced by the real tutorial example module. They must not invent benchmark values, solver traces, or robot configurations.

### Visual language

- Match the Editorial Robotics Lab system: graphite/silver surfaces, restrained teal as the active signal, and no rainbow scientific palette.
- Use clear axes, units, legends, and mathematical labels. Scientific information takes precedence over dramatic camera motion.
- Keep backgrounds compatible with both documentation themes by placing every scene in a deliberate framed plot surface.
- Use direct transitions, opacity, and simple geometric transforms. Avoid particles, elastic easing, decorative 3D sweeps, or looping flourishes.
- Keep each animation short enough to explain one state change and loop without a distracting discontinuity.

### Reproducible rendering

Add:

- `docs/manim/manim.cfg` for resolution, frame rate, background, and media settings;
- `docs/manim/render_kinematics.py` as the single documented render entry point;
- `docs/manim/requirements.txt` with the supported pinned Manim toolchain.

The render script generates both animated GIF and matching static PNG outputs for every scene. Outputs are normalized to deterministic filenames under:

`docs/source/_static/tutorials/kinematics/`

The regular Sphinx requirements and Read the Docs build do not install Manim. Generated outputs are committed, so a docs build consumes assets without regenerating them. The render script validates expected outputs and fails clearly if the isolated Manim environment is unavailable or incomplete.

### Format and motion behavior

- Interactive HTML uses a `<picture>` pattern whose reduced-motion media source selects the PNG before the GIF.
- Normal HTML readers receive the GIF with explicit width and height attributes to prevent layout shift.
- ePub and LaTeX/PDF receive the PNG through builder-appropriate Sphinx branches.
- Every asset has descriptive alt text that states the scientific relationship rather than describing colors or decoration.
- The explanation remains complete when images fail to load or animation is disabled.

## Homepage Motion Gallery

The user-approved composition is **Wide lead, calibrated pair**.

### Desktop composition

- The workspace figure spans the full gallery grid and establishes the section's primary idea.
- Joint trajectory and planned pick motion form a lower `7fr / 5fr` pair.
- The two supporting figures begin and end on shared grid lines; captions use the same baseline rhythm.
- Remove the current second-item top margin, third-item bottom margin, and primary two-row span.
- Use one consistent media frame treatment, border radius, caption spacing, and surface color.

### Scientific image treatment

The source assets have different intrinsic ratios and contain axes or robot geometry near their edges. Therefore:

- use contained, not cover-cropped, image rendering;
- preserve complete axes, legends, tick labels, and robot geometry;
- center media within a stable framed surface rather than stretching it;
- allow controlled internal matte space where ratios differ;
- do not overlay labels or captions on moving media.

“Wide lead” means that the primary **figure** spans the available grid, not that its near-square scientific plot is distorted or aggressively cropped into a banner.

### Responsive composition

- At the existing mobile breakpoint, all figures stack in narrative order.
- The full-width lead retains hierarchy through spacing and caption treatment, not a different crop.
- No figure receives one-off offsets at any breakpoint.
- Explicit media dimensions or aspect-ratio containers prevent content shift.

### Existing assets and fallback

The homepage continues using `workspace.gif`, `joint_trajectory.gif`, and `ur5_pick_motion.gif`. This correction does not regenerate them. If static fallbacks are absent, extract and commit representative PNG frames so reduced-motion and print contexts do not depend on an animated GIF.

## Accessibility and Content Standards

- Maintain a logical heading hierarchy and one canonical page title.
- Keep examples copyable as plain ASCII Python; remove non-breaking spaces and typographic operators from code blocks.
- Retain visible keyboard focus and the existing light/dark contrast tokens.
- Captions are concise and parallel; nearby prose carries the full explanation.
- Avoid claims that cannot be traced to current implementation, including nonexistent APIs, automatic warnings, GPU acceleration promises, or hard-coded performance claims.
- External theory links use HTTPS and do not replace the local conceptual explanation.

## Files in Scope

Expected implementation surface:

- `docs/source/tutorials/kinematics_guide.rst`
- `docs/source/tutorials/Kinematics.rst`
- `docs/source/tutorials/index.rst`
- `docs/examples/kinematics_tutorial.py`
- `docs/manim/kinematics_scenes.py`
- `docs/manim/manim.cfg`
- `docs/manim/render_kinematics.py`
- `docs/manim/requirements.txt`
- `docs/source/_static/tutorials/kinematics/*.{gif,png}`
- `docs/source/index.rst`
- `docs/source/_static/custom.css`
- `tests/test_docs_tutorials.py`
- focused additions to `tests/test_docs_design.py` only where the homepage gallery contract belongs there

No notebooks, public package modules, URDF data, unrelated tutorials, or deployment configuration should change.

## Verification

### Tutorial correctness

- Execute the source-of-truth tutorial module against the bundled Panda URDF.
- Assert that the selected arm view has seven revolute joints, then check FK shape, a 6 by 7 Jacobian, finite velocity output, IK success for the documented reachable target, pose residual tolerances, and compliance with the first seven Panda limits. Separately assert the documented processor behavior that the bundled model also exposes its gripper degree of freedom.
- Verify every tutorial `literalinclude` region exists and is nonempty.
- Scan RST code for non-breaking spaces and references to the known nonexistent APIs removed by this redesign.
- Confirm the tutorials toctree lists only the canonical kinematics page while both old and new HTML routes build.

### Media contracts

- Run the Manim render command in its isolated environment and verify all six expected artifacts exist, have nonzero size, and match the documented dimensions.
- Confirm each GIF has multiple frames and each PNG is readable.
- Confirm animated and still variants use the same scene bounds and scientific labels.
- Verify all HTML animated figures provide a reduced-motion PNG source and explicit dimensions.
- Verify ePub and LaTeX builders reference static PNGs rather than GIFs.

### Homepage gallery

- Add source-level tests that prohibit the former one-off offsets and cover cropping in the gallery.
- In real Chromium, inspect desktop, tablet, and mobile widths in light and dark themes.
- Confirm the lead spans the grid, the lower pair shares a baseline, no axes or labels are clipped, captions align, and the page has no horizontal overflow.
- Confirm `prefers-reduced-motion: reduce` yields static imagery and no reveal motion.

### Documentation builds

- Run focused tutorial/design tests.
- Build HTML without new warnings.
- Build ePub and LaTeX/PDF far enough to validate static media selection.
- Check internal links and image paths.
- Run `git diff --check` and confirm unrelated dirty files remain untouched.
- Apply the `design-taste-frontend` pre-flight checklist to the final rendered tutorial and homepage section.

## Acceptance Criteria

The work is complete when a new reader can follow one Panda-based tutorial from installation link to verified IK result using only APIs that exist, the same code is covered by tests, every Manim study has animated and static accessible forms, the legacy URL still resolves without duplicating content, and the homepage motion gallery reads as one wide lead study over one precisely aligned supporting pair with no cropped scientific information.
