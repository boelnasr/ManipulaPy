# ManipulaPy Notebook Curriculum — Design Spec

**Date:** 2026-06-06
**Branch:** `notebooks/tutorials`
**Status:** Approved (design), pending implementation plan

## 1. Goal

A sequenced set of Jupyter notebooks that **teach robotics through ManipulaPy** — not a
port of the existing `Examples/` scripts. Each notebook explains a concept
intuitively (light derivation, a few key equations), then demonstrates it live with the
library on a consistent robot, with real executed outputs and plots committed to the repo.

**Audience & depth:** Mixed — concept + code, light derivation. Assumes linear algebra;
teaches the robotics. Not a full textbook proof treatment, not an API-only cookbook.

## 2. Running robot

**Franka Emika Panda (7-DOF) everywhere.** Bundled URDF:
`ManipulaPy/ManipulaPy_data/franka_panda/panda.urdf`.

**Known quirk to handle once, up front:** the Panda URDF has 7 revolute joints plus a
fixed `panda_joint8` flange (and hand/fingers). ManipulaPy's screw list comes out **8
columns** while `forward_kinematics` accepts a **7-vector**. The joint convention is
established explicitly in `_shared/helpers.py` and explained in notebook 02 so it is never
a silent off-by-one later.

## 3. Location & structure

`notebooks/` at the repo root:

```
notebooks/
  README.md                          course overview, setup, recommended order
  _shared/helpers.py                 load_panda(), plotting utils, shared configs
  01_rigid_body_motions.ipynb
  02_forward_kinematics.ipynb
  03_velocity_kinematics_jacobians.ipynb
  04_inverse_kinematics.ipynb
  05_dynamics.ipynb
  06_trajectory_planning.ipynb
  07_control.ipynb
  08_singularities_manipulability.ipynb
  09_simulation_pybullet.ipynb
  10_perception_collision.ipynb
  11_capstone_pick_and_place.ipynb
```

## 4. Per-notebook outline

Every notebook follows the same arc: **concept → light math → ManipulaPy demo → plot →
"try it" exercise.**

| # | Topic | Primary ManipulaPy surface |
|---|-------|----------------------------|
| 01 | **Foundations, from the ground up:** rotation matrices & SO(3), angular velocity, exponential/log coordinates for rotation → screw axes, twists, SE(3), exponential coordinates for rigid motion, adjoint maps. The conceptual bedrock the whole course builds on. | `utils`: `MatrixExp3`/`MatrixLog3`, `MatrixExp6`/`MatrixLog6`, `adjoint_transform`, `VecToso3`/`VecTose3`, `se3ToVec`, `TransInv`, `TransToRp` |
| 02 | Forward kinematics (PoE); the 7-vs-8 joint convention; space vs body frames | `SerialManipulator.forward_kinematics` |
| 03 | Jacobians, velocity mapping, 7-DOF redundancy & null space | `jacobian`, `end_effector_velocity` |
| 04 | Inverse kinematics: DLS, smart/SQP, redundancy resolution | `iterative_inverse_kinematics`, `smart_inverse_kinematics` |
| 05 | Dynamics: mass matrix, Coriolis, gravity, inverse/forward dynamics | `ManipulatorDynamics` |
| 06 | Trajectory planning: cubic/quintic, time scaling, joint vs Cartesian | `OptimizedTrajectoryPlanning.joint_trajectory` |
| 07 | Control: PID, computed-torque, adaptive; tracking error analysis | `ManipulatorController` |
| 08 | Singularities, manipulability ellipsoid, workspace analysis | `singularity` module |
| 09 | Simulation in PyBullet: load, step, trajectory playback | `sim.Simulation` |
| 10 | Perception: depth → obstacles, DBSCAN clustering, potential-field collision avoidance; YOLO vision | `perception`, `potential_field`, `vision` |
| 11 | **Capstone** — pick-and-place: perception → IK → trajectory → control → simulation, end-to-end | all of the above |

## 5. Execution strategy

- **Execution environment: system python** (chosen for working cupy / numba-CUDA GPU
  acceleration). numba-CUDA verified operational (1 device, 30 SMs). pybullet, cv2,
  sklearn, matplotlib all work here.
- **torch fix required (this env):** system `torch 2.2.2+cu118` is compiled against NumPy
  1.x but the env runs NumPy 2.2.6 → `RuntimeError: Numpy is not available`. This blocks
  the `[vision]`/`[ml]` path (notebook 10 YOLO, capstone if it uses vision). **Resolution:
  reinstall a NumPy-2-compatible torch (e.g. `torch>=2.3`, matching CUDA) before executing
  notebooks 10 and 11.** Must succeed before those two are finalized.
- **Full execution incl. extras.** Every notebook executed headless via
  `jupyter nbconvert --to notebook --execute --inplace` and committed **with outputs**.
- Notebooks must run top-to-bottom from a clean kernel; no hidden state, no manual steps.

## 5a. Figures — TikZ

All figures are **TikZ/PGF**, compiled with **lualatex** (verified: `standalone.cls`,
`tikz.sty`, `pgfplots.sty` present; `pdftoppm` + ImageMagick `convert` available).

- **Data plots** (trajectories, tracking error, manipulability ellipsoids, residuals):
  matplotlib **pgf backend** (`matplotlib.use("pgf")`, `pgf.texsystem=lualatex`) →
  save PDF → `pdftoppm` to PNG → embed inline.
- **Conceptual diagrams** (coordinate frames, screw axis ŝ/q/h geometry, twist, PoE chain,
  robot schematic): hand-authored standalone **TikZ** → lualatex → PDF → PNG → embed.
- **TikZ source lives in standalone `.tex` files** under `notebooks/_figures/src/` (each a
  complete document that also compiles on its own in any TeX editor). Notebooks load them
  via `render_tikz_file("_figures/src/<name>.tex")` — keeps notebook cells clean and makes
  figures reusable/editable outside Jupyter.
- Centralized in `_shared/tikz.py`: `setup_pgf()`, `embed_pgf_fig(fig, name)`,
  `render_tikz_file(tex_path, name)` (compiles a standalone `.tex` via lualatex → PNG),
  and `render_tikz(code, name)` (same, for quick inline bodies). Returns an
  `IPython.display.Image`. Rendered PNGs live in `notebooks/_figures/` (committed);
  `.tex` *source* is committed under `src/`; intermediate `.pdf`/aux files are gitignored.
  Note: the standalone preamble must include `amsmath` (for `bmatrix`).

## 6. Shared scaffolding

- `_shared/helpers.py`: single source of truth for `load_panda()` (returns
  `SerialManipulator` + `ManipulatorDynamics`) and the Panda joint convention.
- `_shared/tikz.py`: figure helpers (`setup_pgf`, `embed_pgf_fig`, `render_tikz`) — see §5a.
- `notebooks/README.md`: what the course is, prerequisites, install (`pip install -e
  ".[simulation,urdf,vision,ml,cuda]"` + the torch note + a TeX/lualatex note), recommended
  order, and a one-line summary of each notebook.
- Consistent imports across all notebooks; figures are TikZ/PGF (§5a), embedded as PNG.

## 6a. Seed material

A prior UR5 all-in-one notebook (screw theory → FK → dynamics → trajectory) is used as a
**content/style seed**, not kept standalone. Carry forward into notebooks 01–06 on Panda:

- The screw-theory markdown is reusable **after** these corrections:
  - Use one symbol for the twist (`𝒱`/`V`), not `T` then `ξ`.
  - Per-joint screw is `𝒮ᵢ`; the PoE chain must include the home configuration:
    `T = e^{[𝒮₁]θ₁}···e^{[𝒮ₙ]θₙ} · M`.
- Code is rewritten per-notebook on Panda via `_shared/helpers.py`; do **not** reuse the
  seed's broken patterns: hard-coded `"ur5/ur5/ur5.urdf"` path (use the data-dir loader),
  interactive `visualize_robot()` in executed cells (use static plots), and class-level
  `tp.plot_trajectory` (it's an instance method). Old cached outputs are pre-v1.3.2 and
  invalid after the CM-1..CM-6 math fixes — re-execute everything.

## 7. Out of scope (YAGNI)

- Porting the existing `Examples/` scripts (different goal — teaching, not demos).
- A first-party ROS2 package (tracked separately in `ROS2_COMPATIBILITY.md`).
- Real-hardware drivers / `ros2_control`.
- nbsphinx/RTD rendering of the notebooks (can be a later follow-up; not in this scope).

## 8. Success criteria

- All 11 notebooks present under `notebooks/`, each self-contained and runnable from a
  clean kernel.
- Every notebook executed and committed with real outputs/plots (incl. 10 & 11 after the
  torch fix).
- The course is strictly foundations-first: a learner with no robotics background starts
  at screw-axis theory in notebook 01 and each subsequent notebook builds only on concepts
  already introduced — culminating in a working simulated pick-and-place.
- `_shared/helpers.py` and `README.md` present; Panda joint convention handled in one place.
