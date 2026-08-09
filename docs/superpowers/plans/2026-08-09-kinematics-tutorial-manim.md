# Kinematics Tutorial and Manim Studies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one executable, 20-minute Franka Panda kinematics tutorial with tested code, three data-faithful Manim studies, static fallbacks, and a preserved legacy URL.

**Architecture:** A dependency-light Python example module is the source of truth for every displayed calculation and for the numerical data consumed by Manim. The canonical RST page includes marked regions from that module; generated GIF/PNG pairs are committed so Sphinx and Read the Docs never need Manim. Focused tests cover numerical behavior, route ownership, source inclusion, asset integrity, and builder-specific fallback markup.

**Tech Stack:** Python 3.9+, NumPy, ManipulaPy public kinematics/URDF APIs, Sphinx/reStructuredText, Manim Community 0.20.1, Pillow, pytest.

## Global Constraints

- `docs/source/tutorials/kinematics_guide.rst` is the only kinematics tutorial listed in the tutorials toctree.
- `docs/source/tutorials/Kinematics.rst` must continue to build at the legacy URL and must not duplicate examples.
- Use the bundled Franka Panda and a seven-element arm vector; explicitly explain that the processor also exposes one actuated gripper joint.
- Python examples must be plain ASCII, deterministic, executable, and limited to APIs present in the repository.
- The normal docs environment and Read the Docs build must not install or import Manim.
- Each Manim scene must produce a GIF and a matching PNG under `docs/source/_static/tutorials/kinematics/`.
- Reduced-motion HTML, ePub, and LaTeX/PDF must receive PNGs; normal HTML may receive GIFs.
- Do not modify package code, notebooks, URDF data, unrelated tutorials, or deployment configuration.

## File Structure

- Create `docs/examples/kinematics_tutorial.py`: deterministic Panda loading, calculations, residuals, and Manim data.
- Create `tests/test_docs_tutorials.py`: numerical, content, route, media, and builder contracts.
- Rewrite `docs/source/tutorials/kinematics_guide.rst`: canonical narrative and literal includes.
- Rewrite `docs/source/tutorials/Kinematics.rst`: compatibility notice only.
- Modify `docs/source/tutorials/index.rst`: remove the duplicate toctree entry.
- Create `docs/manim/kinematics_scenes.py`: three scientific scenes only.
- Create `docs/manim/manim.cfg`: deterministic 960 by 540, 30 fps rendering defaults.
- Create `docs/manim/render_kinematics.py`: isolated renderer and normalized output copier.
- Create `docs/manim/requirements.txt`: pinned render-only dependency.
- Create six generated assets in `docs/source/_static/tutorials/kinematics/`.

---

### Task 1: Tested Panda tutorial data source

**Files:**
- Create: `docs/examples/kinematics_tutorial.py`
- Create: `tests/test_docs_tutorials.py`

**Interfaces:**
- Produces: `ARM_DOF: int`, `HOME: NDArray`, `TARGET: NDArray`, `JOINT_RATES: NDArray`.
- Produces: `load_panda() -> tuple[SerialManipulator, tuple[str, ...], tuple[tuple[float, float], ...], int]` where the final integer is the processor's full DOF count.
- Produces: `pose_residual(actual, desired) -> tuple[float, float]` in metres and radians.
- Produces: `compute_tutorial_results() -> TutorialResults`.
- Produces: `compute_ik_trace(max_budget: int = 8) -> tuple[NDArray, NDArray, NDArray]` containing iteration budgets, translation residuals, and rotation residuals from real solver calls.

- [ ] **Step 1: Write failing numerical contract tests**

Create `tests/test_docs_tutorials.py` with root/path constants and these initial tests:

```python
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs" / "examples" / "kinematics_tutorial.py"


def load_example():
    spec = importlib.util.spec_from_file_location("kinematics_tutorial", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_panda_arm_contract_and_tutorial_results():
    example = load_example()
    robot, names, limits, full_dof = example.load_panda()
    result = example.compute_tutorial_results()

    assert names == tuple(f"panda_joint{i}" for i in range(1, 8))
    assert len(limits) == example.ARM_DOF == 7
    assert full_dof == 8
    assert result.pose.shape == (4, 4)
    assert result.jacobian.shape == (6, 7)
    assert result.twist.shape == (6,)
    assert np.isfinite(result.twist).all()
    assert result.ik_success
    assert result.ik_iterations <= 20
    assert result.translation_residual < 1e-5
    assert result.rotation_residual < 1e-5
    assert all(low <= q <= high for q, (low, high) in zip(result.ik_solution, limits))
    assert np.allclose(result.pose[3], [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(result.pose[:3, :3].T @ result.pose[:3, :3], np.eye(3), atol=1e-8)


def test_ik_trace_is_real_finite_solver_data():
    example = load_example()
    budgets, translation, rotation = example.compute_ik_trace()
    assert np.array_equal(budgets, np.arange(1, 9))
    assert translation.shape == rotation.shape == budgets.shape
    assert np.isfinite(translation).all()
    assert np.isfinite(rotation).all()
    assert translation[-1] < 1e-5
    assert rotation[-1] < 1e-5
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run: `python3 -m pytest tests/test_docs_tutorials.py -v`

Expected: FAIL because `docs/examples/kinematics_tutorial.py` does not exist.

- [ ] **Step 3: Implement the deterministic example module**

Create the module with this structure and exact tutorial inputs:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ManipulaPy.ManipulaPy_data import get_robot_urdf
from ManipulaPy.urdf.types import JointType
from ManipulaPy.urdf_processor import URDFToSerialManipulator

ARM_DOF = 7
HOME = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.785])
TARGET = np.array([0.15, -0.45, 0.1, -1.8, 0.05, 1.45, 0.65])
JOINT_RATES = np.array([0.05, -0.03, 0.02, 0.0, 0.01, -0.02, 0.03])


@dataclass(frozen=True)
class TutorialResults:
    joint_names: tuple[str, ...]
    pose: NDArray[np.float64]
    jacobian: NDArray[np.float64]
    twist: NDArray[np.float64]
    singular_values: NDArray[np.float64]
    ik_solution: NDArray[np.float64]
    ik_success: bool
    ik_iterations: int
    translation_residual: float
    rotation_residual: float


# [load-panda-start]
def load_panda():
    processor = URDFToSerialManipulator(
        get_robot_urdf("panda"), backend="builtin", load_meshes=False
    )
    arm_joints = tuple(
        joint
        for joint in processor.robot.actuated_joints
        if joint.joint_type is JointType.REVOLUTE and joint.name.startswith("panda_joint")
    )
    if len(arm_joints) != ARM_DOF:
        raise RuntimeError(f"Expected 7 Panda arm joints, found {len(arm_joints)}")
    names = tuple(joint.name for joint in arm_joints)
    if any(joint.limit is None for joint in arm_joints):
        raise RuntimeError("Every Panda arm joint must declare limits")
    limits = tuple((joint.limit.lower, joint.limit.upper) for joint in arm_joints)
    return processor.serial_manipulator, names, limits, processor.robot.num_dofs
# [load-panda-end]


def pose_residual(actual, desired):
    translation = float(np.linalg.norm(actual[:3, 3] - desired[:3, 3]))
    relative_rotation = actual[:3, :3].T @ desired[:3, :3]
    cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
    return translation, float(np.arccos(cosine))
```

Add marked `forward-kinematics`, `velocity-kinematics`, `inverse-kinematics`, and `validation` regions. `compute_tutorial_results()` must call `forward_kinematics(HOME, frame="space")`, `jacobian(HOME, frame="space")`, `end_effector_velocity(HOME, JOINT_RATES, frame="space")`, create the reachable target with `forward_kinematics(TARGET)`, and call:

```python
solution, success, iterations = robot.iterative_inverse_kinematics(
    target_pose,
    HOME,
    max_iterations=400,
    adaptive_tuning=True,
    backtracking=True,
)
```

Implement `compute_ik_trace()` by rerunning that same public solver for budgets `1..max_budget`, measuring each returned pose with `pose_residual`, and returning NumPy arrays. This produces an honest convergence study, including any brief tradeoff between translation and rotation error.

- [ ] **Step 4: Run the numerical contract tests**

Run: `python3 -m pytest tests/test_docs_tutorials.py -v`

Expected: 2 passed. Warnings about unresolved optional mesh URIs are acceptable because `load_meshes=False`; no test may require PyBullet rendering.

- [ ] **Step 5: Run style and syntax checks**

Run: `python3 -m py_compile docs/examples/kinematics_tutorial.py tests/test_docs_tutorials.py`

Run: `python3 -m ruff check docs/examples/kinematics_tutorial.py tests/test_docs_tutorials.py`

Expected: both commands pass.

- [ ] **Step 6: Commit the source-of-truth example**

```bash
git add docs/examples/kinematics_tutorial.py tests/test_docs_tutorials.py
git commit -m "docs: add tested Panda kinematics example"
```

---

### Task 2: Canonical tutorial and compatibility route

**Files:**
- Modify: `docs/source/tutorials/kinematics_guide.rst`
- Modify: `docs/source/tutorials/Kinematics.rst`
- Modify: `docs/source/tutorials/index.rst`
- Modify: `tests/test_docs_tutorials.py`

**Interfaces:**
- Consumes: marker regions and constants from `docs/examples/kinematics_tutorial.py`.
- Produces: canonical route `tutorials/kinematics_guide.html` and compatibility route `tutorials/Kinematics.html`.

- [ ] **Step 1: Add failing content and route tests**

Append tests that assert:

```python
TUTORIALS = ROOT / "docs" / "source" / "tutorials"


def read(path):
    return path.read_text(encoding="utf-8")


def test_one_canonical_kinematics_tutorial_route():
    index = read(TUTORIALS / "index.rst")
    canonical = read(TUTORIALS / "kinematics_guide.rst")
    legacy = read(TUTORIALS / "Kinematics.rst")
    assert "   kinematics_guide\n" in index
    assert "   Kinematics\n" not in index
    assert "Kinematics with the Franka Panda" in canonical
    assert "This tutorial has moved" in legacy
    assert ":doc:`kinematics_guide`" in legacy
    assert "code-block:: python" not in legacy


def test_tutorial_uses_tested_regions_and_current_api():
    source = read(TUTORIALS / "kinematics_guide.rst")
    for marker in (
        "load-panda",
        "forward-kinematics",
        "velocity-kinematics",
        "inverse-kinematics",
        "validation",
    ):
        assert f":start-after: # [{marker}-start]" in source
        assert f":end-before: # [{marker}-end]" in source
    for forbidden in (
        "batch_forward_kinematics",
        "sample_workspace",
        "lm_inverse_kinematics",
        "position_inverse_kinematics",
        "success, sol",
        "Python ≥\u00a03.8",
    ):
        assert forbidden not in source
    assert "\u00a0" not in source
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python3 -m pytest tests/test_docs_tutorials.py -k 'canonical or tested_regions' -v`

Expected: FAIL because the toctree still lists both routes and the canonical page is incomplete.

- [ ] **Step 3: Rewrite the canonical page**

Replace the current page with these sections in order:

1. `Kinematics with the Franka Panda`
2. `What you will build`
3. `Before you begin`
4. `Load the bundled Panda`
5. `Forward kinematics: joints to pose`
6. `Jacobian: joints to tool velocity`
7. `Inverse kinematics: pose to joints`
8. `Validate the result`
9. `Troubleshooting`
10. `Go deeper`

Use `.. literalinclude:: ../../examples/kinematics_tutorial.py` with `:language: python`, `:dedent: 0`, and the exact start/end markers tested above. State Python 3.9+, explain the seven arm joints plus exposed gripper DOF once, label angular twist units as rad/s and linear units as m/s, and explain why pose residuals are checked instead of joint-vector equality.

Add short expected-output blocks for shapes and tolerances, not full frozen matrices. Link to `notebook_course`, `../user_guide/Kinematics`, `../api/kinematics`, `../user_guide/URDF_Processor`, and `../user_guide/Singularity_Analysis`.

- [ ] **Step 4: Convert the legacy page and toctree**

Make `Kinematics.rst` a short compatibility page with the exact heading and link required by the test. Remove only the `Kinematics` line from the tutorials toctree; preserve every other route and its order.

- [ ] **Step 5: Run route/content tests and an HTML build**

Run: `python3 -m pytest tests/test_docs_tutorials.py -v`

Run: `make -C docs html SPHINXSTRICT=1`

Expected: tests pass; Sphinx builds both `docs/build/html/tutorials/kinematics_guide.html` and `docs/build/html/tutorials/Kinematics.html` without new warnings.

- [ ] **Step 6: Commit the canonical tutorial**

```bash
git add docs/source/tutorials/kinematics_guide.rst docs/source/tutorials/Kinematics.rst docs/source/tutorials/index.rst tests/test_docs_tutorials.py
git commit -m "docs: consolidate the Panda kinematics tutorial"
```

---

### Task 3: Reproducible Manim rendering pipeline

**Files:**
- Create: `docs/manim/kinematics_scenes.py`
- Create: `docs/manim/manim.cfg`
- Create: `docs/manim/render_kinematics.py`
- Create: `docs/manim/requirements.txt`
- Modify: `tests/test_docs_tutorials.py`

**Interfaces:**
- Consumes: `HOME`, `TARGET`, `JOINT_RATES`, `compute_tutorial_results()`, and `compute_ik_trace()` from the example module.
- Produces scene classes: `PandaForwardKinematics`, `PandaJacobianVelocity`, `PandaIKConvergence`.
- Produces filenames: `panda_forward_kinematics.gif/.png`, `panda_jacobian_velocity.gif/.png`, `panda_ik_convergence.gif/.png`.

- [ ] **Step 1: Add failing pipeline source tests**

Append tests that parse the render sources without requiring Manim:

```python
MANIM = ROOT / "docs" / "manim"
ASSETS = ROOT / "docs" / "source" / "_static" / "tutorials" / "kinematics"


def test_manim_pipeline_is_render_only_and_pinned():
    scenes = read(MANIM / "kinematics_scenes.py")
    renderer = read(MANIM / "render_kinematics.py")
    requirements = read(MANIM / "requirements.txt")
    assert requirements.strip() == "manim==0.20.1"
    for scene in (
        "PandaForwardKinematics",
        "PandaJacobianVelocity",
        "PandaIKConvergence",
    ):
        assert f"class {scene}" in scenes
        assert scene in renderer
    assert "compute_tutorial_results" in scenes
    assert "compute_ik_trace" in scenes
    assert "docs/requirements.txt" not in renderer


def test_manim_config_has_stable_scientific_output():
    config = read(MANIM / "manim.cfg")
    for contract in ("pixel_width = 960", "pixel_height = 540", "frame_rate = 30"):
        assert contract in config
```

- [ ] **Step 2: Run the pipeline source tests and verify failure**

Run: `python3 -m pytest tests/test_docs_tutorials.py -k manim -v`

Expected: FAIL because `docs/manim` does not exist.

- [ ] **Step 3: Add render-only dependency and configuration**

Create `requirements.txt` containing exactly `manim==0.20.1` plus a final newline. Create `manim.cfg` with `[CLI]`, `pixel_width = 960`, `pixel_height = 540`, `frame_rate = 30`, `background_color = #172126`, `disable_caching = True`, and `verbosity = WARNING`.

- [ ] **Step 4: Implement three restrained scientific scenes**

At module load, add `docs/examples` to `sys.path` relative to `__file__`, then import only the tested example interfaces. Define shared constants `INK = "#E6ECEE"`, `MUTED = "#94A3A8"`, `TEAL = "#63C9BC"`, `RULE = "#405158"`, and `PANEL = "#172126"`.

Implement:

- `PandaForwardKinematics`: a base-to-tool seven-joint chain with numbered joints, a tool-frame triad, and the 4 by 4 transform blocks. Animate one joint-chain reveal and one transform update.
- `PandaJacobianVelocity`: a seven-value joint-rate column, a 6 by 7 Jacobian heat/value matrix, and two labeled output groups, `angular [rad/s]` and `linear [m/s]`. Use the actual result arrays.
- `PandaIKConvergence`: two axes sharing iteration on x but using separately labeled residual scales. Plot the real arrays from `compute_ik_trace()` and mark the solved tolerance. Do not force the trace to look monotonic.

Every scene must use `Text` for prose, `MathTex` only for mathematical notation, at most one teal emphasis channel, linear or smooth easing, and no camera rotation, particles, bounce, or elastic motion.

- [ ] **Step 5: Implement the normalized renderer**

The script must:

- resolve repository, scene, config, and final asset paths from `Path(__file__)`;
- accept `--scene {all,fk,jacobian,ik}` with `all` as default;
- accept `--check` to validate committed outputs without importing Manim;
- render each selected scene twice in a `TemporaryDirectory`, once with `--format=gif` and once with `--format=png --save_last_frame`;
- find exactly one matching file for each render and copy it to the deterministic final filename;
- validate `.gif` and `.png` suffixes, nonzero sizes, and Pillow-reported `(960, 540)` dimensions;
- exit nonzero with a precise missing/ambiguous-output message.

Use `subprocess.run([...], check=True)` argument lists; do not construct shell command strings.

- [ ] **Step 6: Run source tests and syntax checks**

Run: `python3 -m pytest tests/test_docs_tutorials.py -k manim -v`

Run: `python3 -m py_compile docs/manim/kinematics_scenes.py docs/manim/render_kinematics.py`

Expected: all pass without importing Manim from the regular pytest process.

- [ ] **Step 7: Commit the rendering pipeline**

```bash
git add docs/manim tests/test_docs_tutorials.py
git commit -m "docs: add Manim kinematics render pipeline"
```

---

### Task 4: Render and validate the six scientific assets

**Files:**
- Create: `docs/source/_static/tutorials/kinematics/panda_forward_kinematics.gif`
- Create: `docs/source/_static/tutorials/kinematics/panda_forward_kinematics.png`
- Create: `docs/source/_static/tutorials/kinematics/panda_jacobian_velocity.gif`
- Create: `docs/source/_static/tutorials/kinematics/panda_jacobian_velocity.png`
- Create: `docs/source/_static/tutorials/kinematics/panda_ik_convergence.gif`
- Create: `docs/source/_static/tutorials/kinematics/panda_ik_convergence.png`
- Modify: `tests/test_docs_tutorials.py`

**Interfaces:**
- Consumes: the Task 3 renderer.
- Produces: six committed Sphinx-ready assets with identical scene bounds.

- [ ] **Step 1: Add failing committed-asset tests**

```python
from PIL import Image


def test_committed_kinematics_media_pairs_are_valid():
    stems = (
        "panda_forward_kinematics",
        "panda_jacobian_velocity",
        "panda_ik_convergence",
    )
    for stem in stems:
        gif = ASSETS / f"{stem}.gif"
        png = ASSETS / f"{stem}.png"
        assert gif.stat().st_size > 10_000
        assert png.stat().st_size > 10_000
        with Image.open(gif) as animated:
            assert animated.size == (960, 540)
            assert getattr(animated, "n_frames", 1) > 1
        with Image.open(png) as still:
            assert still.size == (960, 540)
            assert still.format == "PNG"
```

- [ ] **Step 2: Run the asset test and verify failure**

Run: `python3 -m pytest tests/test_docs_tutorials.py::test_committed_kinematics_media_pairs_are_valid -v`

Expected: FAIL because the committed assets do not exist.

- [ ] **Step 3: Create an isolated Manim environment**

Run:

```bash
python3 -m venv /tmp/manipulapy-manim-venv
/tmp/manipulapy-manim-venv/bin/python -m pip install -r docs/manim/requirements.txt
```

If system Cairo/Pango/FFmpeg prerequisites are missing, install the packages named by Manim's error through the platform package manager, then rerun the same pip command. Do not add Manim to project or docs runtime dependencies.

- [ ] **Step 4: Render every scene and inspect output**

Run: `/tmp/manipulapy-manim-venv/bin/python docs/manim/render_kinematics.py --scene all`

Run: `python3 docs/manim/render_kinematics.py --check`

Expected: six outputs pass size, format, frame-count, and dimension checks.

Open or inspect all six assets. Confirm axes, units, legends, matrices, and robot/joint labels are fully visible at actual output size, and that the final PNG matches the GIF's final explanatory state.

- [ ] **Step 5: Run asset tests**

Run: `python3 -m pytest tests/test_docs_tutorials.py::test_committed_kinematics_media_pairs_are_valid -v`

Expected: PASS.

- [ ] **Step 6: Commit generated scientific media**

```bash
git add docs/source/_static/tutorials/kinematics tests/test_docs_tutorials.py
git commit -m "docs: add Panda kinematics motion studies"
```

---

### Task 5: Embed accessible animated/static studies

**Files:**
- Modify: `docs/source/tutorials/kinematics_guide.rst`
- Modify: `docs/source/_static/custom.css`
- Modify: `tests/test_docs_tutorials.py`

**Interfaces:**
- Consumes: all six Task 4 assets.
- Produces: GIF-first normal HTML, PNG-first reduced-motion HTML, and PNG-only ePub/LaTeX content.

- [ ] **Step 1: Add failing media-embedding tests**

Add a table of stem/alt-text pairs and assert each appears in the canonical source. For every stem, require:

```python
assert f'<source media="(prefers-reduced-motion: reduce)" srcset="../_static/tutorials/kinematics/{stem}.png">' in source
assert f'<img src="../_static/tutorials/kinematics/{stem}.gif" width="960" height="540"' in source
assert f".. image:: ../_static/tutorials/kinematics/{stem}.png" in source
```

Also assert the source contains `.. only:: html and not epub`, `.. only:: epub`, and `.. only:: latex` branches, and that every `<img>` has nonempty `alt` and `loading="lazy"` attributes.

- [ ] **Step 2: Run the embedding test and verify failure**

Run: `python3 -m pytest tests/test_docs_tutorials.py -k embedding -v`

Expected: FAIL because the media is not yet referenced.

- [ ] **Step 3: Add each study beside its corresponding explanation**

After FK, Jacobian/velocity, and IK respectively, add:

- an HTML-only `<figure class="mp-tutorial-study">` containing `<picture>`, a reduced-motion PNG `<source>`, the GIF `<img>`, and a concise `<figcaption>`;
- an ePub-only `.. figure::` using the PNG;
- a LaTeX-only `.. figure::` using the PNG.

Use these alt-text meanings:

- FK: “Seven Panda arm joints build a base-to-tool transform and reveal the resulting tool frame.”
- velocity: “Seven joint rates pass through a six-by-seven Jacobian into angular and linear tool velocity.”
- IK: “Translation and rotation residuals converge as inverse kinematics approaches a reachable Panda pose.”

Do not overlay prose on the animation. Keep the complete explanation in surrounding RST.

- [ ] **Step 4: Add restrained tutorial-study CSS**

Add a small `.mp-tutorial-study` block to `docs/source/_static/custom.css` using `margin-block: 2rem`, `background: var(--mp-panel-strong)`, `border: 1px solid var(--mp-rule)`, `border-radius: var(--mp-radius)`, and `overflow: hidden`. Set its `picture` and `img` to `display: block`, its image to `width: 100%; height: auto;`, and its caption to the existing mono/muted caption treatment. Do not add a new color, shadow, or animation system.

- [ ] **Step 5: Run focused tests and all three Sphinx formats**

Run: `python3 -m pytest tests/test_docs_tutorials.py -v`

Run: `make -C docs html SPHINXSTRICT=1`

Run: `make -C docs epub SPHINXSTRICT=1`

Run: `make -C docs latex SPHINXSTRICT=1`

Inspect generated HTML for `.gif` plus reduced-motion `.png`; inspect the built EPUB archive and generated `.tex` for PNG references and absence of kinematics GIF references.

- [ ] **Step 6: Commit accessible media integration**

```bash
git add docs/source/tutorials/kinematics_guide.rst docs/source/_static/custom.css tests/test_docs_tutorials.py
git commit -m "docs: embed accessible kinematics studies"
```

---

### Task 6: Final tutorial QA and regression gate

**Files:**
- Modify only files from Tasks 1-5 when a verified defect requires correction.

**Interfaces:**
- Consumes: completed canonical tutorial, compatibility route, renderer, assets, and tests.
- Produces: verified tutorial deliverable with no unrelated changes.

- [ ] **Step 1: Run the complete focused test set**

Run: `python3 -m pytest tests/test_docs_tutorials.py tests/test_docs_design.py -v`

Expected: all tests pass.

- [ ] **Step 2: Run executable and mechanical checks**

Run: `python3 docs/examples/kinematics_tutorial.py`

Run: `python3 docs/manim/render_kinematics.py --check`

Run: `rg -n $'\u00a0|batch_forward_kinematics|sample_workspace|lm_inverse_kinematics|position_inverse_kinematics' docs/source/tutorials/kinematics_guide.rst docs/source/tutorials/Kinematics.rst`

Expected: example exits zero, media check passes, and the scan returns no matches.

- [ ] **Step 3: Build and inspect documentation**

Run HTML, ePub, and LaTeX commands from Task 5. Serve HTML locally and inspect the canonical and compatibility routes at desktop and mobile widths in light and dark themes. Verify code copy, figure containment, caption rhythm, keyboard focus, no horizontal overflow, and reduced-motion PNG selection.

- [ ] **Step 4: Run final repository checks**

Run: `git diff --check`

Run: `git status --short`

Expected: no whitespace errors; only explicitly preserved pre-existing untracked files may remain.

- [ ] **Step 5: Commit verified corrections if any**

If Step 1-4 exposed an in-scope defect, stage only the exact corrected files and commit:

```bash
git commit -m "fix: complete kinematics tutorial verification"
```

If no corrections were needed, do not create an empty commit.
