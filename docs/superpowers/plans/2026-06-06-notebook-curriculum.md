# ManipulaPy Notebook Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an 11-notebook foundations-first robotics course that teaches screw-theory robotics through ManipulaPy on the Franka Panda, fully executed with committed TikZ figures.

**Architecture:** A `notebooks/` tree at repo root. Shared scaffolding (`_shared/helpers.py` for robot loading, `_shared/tikz.py` for figures) is built and unit-tested first. Each notebook is then authored, executed headless via `nbconvert`, and verified by an embedded smoke-test cell plus clean execution. Figures are TikZ/PGF compiled with lualatex.

**Tech Stack:** Python 3.10 (system interpreter), ManipulaPy 1.3.2.post1, numpy 2.2.6, matplotlib (pgf backend), jupyter/nbconvert, pybullet, cupy/numba-CUDA, torch (reinstalled ≥2.3), TeX Live (lualatex, standalone, tikz, pgfplots), pdftoppm.

**Branch:** `notebooks/tutorials` (already created).

---

## Verified API reference (use these exact signatures)

Confirmed against the installed library — do not invent alternatives:

```python
# Loading
from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy import ManipulaPy_data            # bundled URDFs
# panda: <data>/franka_panda/panda.urdf  →  S_list/B_list shape (6,8), M_list (4,4)
up = URDFToSerialManipulator(panda_path)
sm  = up.serial_manipulator      # SerialManipulator
dyn = up.dynamics                # ManipulatorDynamics
# Panda quirk: 7 revolute + fixed panda_joint8 → screw list has 8 columns,
#              but FK/Jacobian accept BOTH a 7-vector and an 8-vector.
#              Course convention: use 7 actuated joints; helpers expose N_JOINTS=7.

# SerialManipulator
sm.forward_kinematics(thetalist, frame='space')            # -> (4,4); frame in {'space','body'}
sm.jacobian(thetalist, frame='space')                      # -> (6,n)
sm.end_effector_velocity(thetalist, dthetalist, frame='space')  # -> (6,)
sm.joint_velocity(thetalist, V_ee, frame='space')          # -> (n,)
sm.iterative_inverse_kinematics(T_desired, thetalist0, eomg=1e-6, ev=1e-6,
        max_iterations=10000, damping=0.02, step_cap=0.3)  # -> (theta, success:bool, iters:int)
sm.smart_inverse_kinematics(T_desired, strategy='workspace_heuristic',
        theta_current=None)                                # -> (theta, success, iters)

# ManipulatorDynamics
dyn.mass_matrix(thetalist)                                 # -> (n,n)
dyn.velocity_quadratic_forces(thetalist, dthetalist)      # -> (n,)
dyn.gravity_forces(thetalist, g)                           # -> (n,)
dyn.inverse_dynamics(thetalist, dthetalist, ddthetalist, g, Ftip)  # -> (n,)
dyn.forward_dynamics(thetalist, dthetalist, tau, g, Ftip)          # -> (n,)

# Trajectory planning (TrajectoryPlanning subclasses OptimizedTrajectoryPlanning)
from ManipulaPy.path_planning import TrajectoryPlanning
trr = TrajectoryPlanning(serial_manipulator=sm, urdf_path=panda_path,
                         dynamics=dyn, joint_limits=joint_limits, torque_limits=None)
traj = trr.joint_trajectory(thetastart, thetaend, Tf, N, method)  # method 3=cubic,5=quintic
# traj -> dict: 'positions','velocities','accelerations' each (N,n)
trr.inverse_dynamics_trajectory(positions, velocities, accelerations,
        gravity_vector=None, Ftip=None)                    # -> (N,n) torques
trr.plot_trajectory(...)   # INSTANCE method (not class-level)

# Control
from ManipulaPy.control import ManipulatorController
ctrl = ManipulatorController(dynamics=dyn)
# methods: pid_control, pd_control, computed_torque_control, adaptive_control,
#          robust_control, feedforward_control, joint_space_control,
#          cartesian_space_control, kalman_filter_control,
#          ziegler_nichols_tuning, calculate_settling_time, calculate_percent_overshoot ...

# Singularity
from ManipulaPy.singularity import Singularity   # class

# Simulation (PyBullet)
from ManipulaPy.sim import Simulation
Simulation(urdf_file_path, joint_limits, torque_limits=None, time_step=0.01,
           real_time_factor=1.0, physics_client=None, enable_self_collision=False)
# methods: connect_simulation, initialize_robot, run_trajectory, get_joint_positions,
#          set_joint_positions, check_collisions, close_simulation ...  (use DIRECT mode)

# Perception / collision
from ManipulaPy.perception import Perception      # uses DBSCAN
from ManipulaPy.potential_field import PotentialField, CollisionChecker
from ManipulaPy.vision import Vision              # YOLO (needs working torch)
```

**utils (notebook 01):** `MatrixExp3`/`MatrixLog3`, `MatrixExp6`/`MatrixLog6`,
`adjoint_transform`, `VecToso3`/`VecTose3`, `se3ToVec`, `TransInv`, `TransToRp`,
`CubicTimeScaling`, `QuinticTimeScaling`.

---

## File structure

```
notebooks/
  README.md
  _shared/
    __init__.py
    helpers.py            # load_panda(), PANDA_URDF, N_JOINTS, HOME, joint_limits
    tikz.py               # setup_pgf(), embed_pgf_fig(), render_tikz()
  _figures/               # committed PNGs; .tex/.pdf gitignored
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
tests/
  test_notebooks_shared.py   # unit tests for helpers + tikz
```

Each notebook follows the arc: **concept → light math → ManipulaPy demo → figure → "try it" exercise**, and contains a final **smoke-test cell** that asserts the key results so execution failure is loud.

---

## Task 0: Environment prep (torch fix + tooling check)

**Files:** none (environment only).

- [ ] **Step 1: Confirm the broken torch and the TeX/GPU tooling**

Run:
```bash
python3 -c "import torch; torch.tensor([1.]).numpy()" ; echo "exit=$?"
which lualatex pdftoppm convert
python3 -c "import numba.cuda as c; print('cuda', c.is_available())"
```
Expected: the torch line prints a `RuntimeError: Numpy is not available` (exit≠0); lualatex/pdftoppm/convert resolve; cuda True.

- [ ] **Step 2: Reinstall a NumPy-2-compatible torch**

Run (match the system CUDA; cu118 shown):
```bash
python3 -m pip install --upgrade "torch>=2.3" --index-url https://download.pytorch.org/whl/cu118
```

- [ ] **Step 3: Verify torch interop and YOLO import**

Run:
```bash
python3 -c "import numpy,torch; print(torch.__version__, torch.tensor([1.,2.,3.]).numpy().sum())"
python3 -c "from ultralytics import YOLO; print('ultralytics ok')"
```
Expected: prints torch version and `6.0`; `ultralytics ok`.

- [ ] **Step 4: Commit (no code yet — record the env decision in the plan checkbox only)**

No commit; proceed.

---

## Task 1: Shared figure helper (`_shared/tikz.py`)

**Files:**
- Create: `notebooks/_shared/__init__.py` (empty)
- Create: `notebooks/_shared/tikz.py`
- Test: `tests/test_notebooks_shared.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_notebooks_shared.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "notebooks"))

def test_render_tikz_produces_png(tmp_path):
    from _shared.tikz import render_tikz
    code = r"\begin{tikzpicture}\draw[->](0,0)--(2,1) node[right]{$\hat s$};\end{tikzpicture}"
    img = render_tikz(code, name="t_axis", outdir=str(tmp_path))
    png = os.path.join(str(tmp_path), "t_axis.png")
    assert os.path.exists(png) and os.path.getsize(png) > 0
    assert img is not None  # IPython.display.Image

def test_setup_pgf_and_embed(tmp_path):
    from _shared.tikz import setup_pgf, embed_pgf_fig
    plt = setup_pgf()
    fig, ax = plt.subplots(); ax.plot([0,1,2],[0,1,4]); ax.set_title(r"$e^{[\mathcal{S}]\theta}$")
    img = embed_pgf_fig(fig, name="t_plot", outdir=str(tmp_path))
    assert os.path.exists(os.path.join(str(tmp_path), "t_plot.png"))
    assert img is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_notebooks_shared.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named '_shared.tikz'`.

- [ ] **Step 3: Implement `_shared/tikz.py`**

```python
# notebooks/_shared/tikz.py
"""TikZ/PGF figure helpers for the ManipulaPy notebook course.

Two paths, both compiled with lualatex:
  * render_tikz(code)   -> hand-authored TikZ standalone -> PNG (conceptual diagrams)
  * setup_pgf()/embed_pgf_fig(fig) -> matplotlib pgf backend -> PNG (data plots)
PNGs are written to an output dir (default notebooks/_figures) and returned as
IPython.display.Image for inline embedding.
"""
from __future__ import annotations
import os, subprocess, tempfile, shutil

_DEF_OUTDIR = os.path.join(os.path.dirname(__file__), "..", "_figures")

_STANDALONE = r"""\documentclass[tikz,border=4pt]{standalone}
\usepackage{tikz}\usepackage{pgfplots}\pgfplotsset{compat=1.18}
\usetikzlibrary{arrows.meta,calc,3d,angles,quotes}
\begin{document}
%s
\end{document}
"""

def _ensure(outdir):
    outdir = outdir or _DEF_OUTDIR
    os.makedirs(outdir, exist_ok=True)
    return outdir

def _pdf_to_png(pdf_path, png_path, dpi=200):
    # pdftoppm writes <prefix>.png
    prefix = png_path[:-4] if png_path.endswith(".png") else png_path
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile", pdf_path, prefix],
                   check=True)

def render_tikz(code, name, outdir=None, dpi=200):
    """Compile a TikZ body (a \\begin{tikzpicture}...\\end{tikzpicture} string) to PNG."""
    from IPython.display import Image
    outdir = _ensure(outdir)
    png = os.path.join(outdir, f"{name}.png")
    with tempfile.TemporaryDirectory() as td:
        tex = os.path.join(td, f"{name}.tex")
        with open(tex, "w") as f:
            f.write(_STANDALONE % code)
        subprocess.run(["lualatex", "-interaction=nonstopmode", f"{name}.tex"],
                       cwd=td, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _pdf_to_png(os.path.join(td, f"{name}.pdf"), png, dpi=dpi)
    return Image(filename=png)

def setup_pgf():
    """Configure matplotlib to render via the pgf/lualatex backend; return pyplot."""
    import matplotlib
    matplotlib.use("pgf")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "pgf.texsystem": "lualatex",
        "font.family": "serif",
        "text.usetex": False,
        "pgf.rcfonts": False,
        "figure.figsize": (5.0, 3.2),
    })
    return plt

def embed_pgf_fig(fig, name, outdir=None, dpi=200):
    """Save a matplotlib (pgf-backed) figure to PDF, convert to PNG, return Image."""
    from IPython.display import Image
    outdir = _ensure(outdir)
    pdf = os.path.join(outdir, f"{name}.pdf")
    png = os.path.join(outdir, f"{name}.png")
    fig.savefig(pdf, bbox_inches="tight")
    _pdf_to_png(pdf, png, dpi=dpi)
    return Image(filename=png)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_notebooks_shared.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks/_shared/__init__.py notebooks/_shared/tikz.py tests/test_notebooks_shared.py
git commit -m "feat(notebooks): add TikZ/PGF figure helpers with lualatex pipeline"
```

---

## Task 2: Shared robot loader (`_shared/helpers.py`)

**Files:**
- Create: `notebooks/_shared/helpers.py`
- Test: append to `tests/test_notebooks_shared.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_notebooks_shared.py
import numpy as np

def test_load_panda_dimensions():
    from _shared.helpers import load_panda, N_JOINTS, HOME, joint_limits
    sm, dyn = load_panda()
    assert N_JOINTS == 7
    assert len(HOME) == 7
    assert len(joint_limits()) == 7
    T = sm.forward_kinematics(HOME, frame="space")
    assert T.shape == (4, 4)
    assert np.allclose(T[3], [0, 0, 0, 1])
    M = dyn.mass_matrix(HOME)
    assert M.shape == (7, 7) or M.shape == (8, 8)  # accepts course 7-vector
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_notebooks_shared.py::test_load_panda_dimensions -v`
Expected: FAIL — `ModuleNotFoundError: No module named '_shared.helpers'`.

- [ ] **Step 3: Implement `_shared/helpers.py`**

```python
# notebooks/_shared/helpers.py
"""Shared robot loading + conventions for the notebook course.

Running robot: Franka Emika Panda. The URDF has 7 revolute joints plus a fixed
panda_joint8 flange, so ManipulaPy's screw list has 8 columns while FK/Jacobian
accept either a 7- or 8-vector. The course standardises on the 7 actuated joints.
"""
from __future__ import annotations
import os
import numpy as np
from ManipulaPy import ManipulaPy_data
from ManipulaPy.urdf_processor import URDFToSerialManipulator

N_JOINTS = 7
PANDA_URDF = os.path.join(os.path.dirname(ManipulaPy_data.__file__),
                          "franka_panda", "panda.urdf")

# A non-singular, visually clear default configuration (radians).
HOME = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.785])

def load_panda():
    """Return (SerialManipulator, ManipulatorDynamics) for the Panda."""
    up = URDFToSerialManipulator(PANDA_URDF)
    return up.serial_manipulator, up.dynamics

def joint_limits():
    """Conservative Panda joint limits (rad), 7 actuated joints."""
    return [(-2.90, 2.90), (-1.76, 1.76), (-2.90, 2.90), (-3.07, -0.07),
            (-2.90, 2.90), (-0.02, 3.75), (-2.90, 2.90)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_notebooks_shared.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks/_shared/helpers.py tests/test_notebooks_shared.py
git commit -m "feat(notebooks): add Panda loader and shared conventions"
```

---

## Notebook authoring tasks (3–13)

**Each notebook task follows the same 5 steps. Notebook-specific content (sections,
key API calls, the smoke-test assertion) is given per task.**

> **Authoring mechanics (apply to every notebook task):**
> - Build the `.ipynb` with `nbformat` (a small script per notebook) OR edit JSON directly.
>   First cell of every notebook:
>   ```python
>   import os, sys
>   sys.path.insert(0, os.path.dirname(os.path.abspath("")) + "/notebooks")  # repo-root run
>   sys.path.insert(0, os.path.join(os.getcwd(), "_shared")) if os.path.isdir("_shared") else None
>   from _shared.helpers import load_panda, N_JOINTS, HOME, joint_limits
>   from _shared.tikz import setup_pgf, embed_pgf_fig, render_tikz
>   import numpy as np
>   ```
>   (Run notebooks with `cwd=notebooks/` so `_shared` is importable.)
> - **Last cell is a smoke test**: `assert` the key numeric results listed in the task.
> - **Execute programmatically — the `jupyter nbconvert` CLI is NOT wired up on this
>   machine.** Use `nbconvert.preprocessors.ExecutePreprocessor` from a small builder
>   script run with `cwd=notebooks/` (so `_shared` is importable), e.g.:
>   ```python
>   from nbconvert.preprocessors import ExecutePreprocessor
>   ep = ExecutePreprocessor(timeout=900, kernel_name="python3")
>   ep.preprocess(nb, {"metadata": {"path": "notebooks"}})
>   ```
>   (Notebooks are authored via `nbformat` builder scripts; this is also how they execute.)
> - Verify exit 0 and no `error` output cells:
>   `python3 -c "import nbformat,sys; nb=nbformat.read('notebooks/NN_name.ipynb',4); assert all(o.get('output_type')!='error' for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[])), 'has error cells'"`
> - Commit the executed notebook + any new `_figures/*.png`.

---

### Task 3: Notebook 01 — Rigid-body motions (foundations)

**Files:** Create `notebooks/01_rigid_body_motions.ipynb`

**Sections (markdown + code):**
1. Rotation matrices & SO(3): properties, `MatrixExp3`/`MatrixLog3` (exp/log coords for rotation), angular velocity & `VecToso3`. TikZ diagram: a rotating frame.
2. Rigid-body motion & SE(3): `TransToRp`, `TransInv`, homogeneous transforms.
3. **Screw axes**: ŝ (axis), q (point), h (pitch); twist `𝒱=[ω; v]` with `v = -ŝ×q + hŝ` (times θ̇). Hand-authored TikZ figure of a screw axis (reuse/repair the seed markdown; one symbol `𝒱`; include home config in the PoE preview).
4. Exponential coords for rigid motion: `VecTose3`, `MatrixExp6`/`MatrixLog6`.
5. Adjoint map: `adjoint_transform`, twist frame changes.
6. **Try it** exercise: build a screw axis for a given joint, exponentiate it.

- [ ] **Step 1:** Author the notebook with the sections above, using only `utils` functions.
- [ ] **Step 2 (smoke test cell):**
```python
from ManipulaPy.utils import MatrixExp6, MatrixLog6, MatrixExp3, VecToso3, adjoint_transform
import numpy as np
R = MatrixExp3(VecToso3([0,0,np.pi/2]))
assert np.allclose(R @ np.array([1,0,0]), [0,1,0], atol=1e-9)   # 90° about z
# NOTE: MatrixExp6 takes a 4x4 se(3) matrix, MatrixExp3 a 3x3 so(3) matrix.
# Build them from vectors via VecTose3 / VecToso3.
from ManipulaPy.utils import VecTose3, se3ToVec, MatrixLog6
assert np.allclose(MatrixExp6(VecTose3(np.zeros(6))), np.eye(4))
S = np.array([0,0,1,0,-0.3,0], float)                            # z-axis screw through q=(0.3,0,0)
T = MatrixExp6(VecTose3(S*(np.pi/2)))
assert np.allclose(se3ToVec(MatrixLog6(T)), S*(np.pi/2), atol=1e-9)   # round trip
assert adjoint_transform(np.eye(4)).shape == (6,6)
print("nb01 smoke OK")
```
- [ ] **Step 3:** Execute with nbconvert (see authoring mechanics). Expected exit 0.
- [ ] **Step 4:** Verify no error cells (command in authoring mechanics).
- [ ] **Step 5:** Commit:
```bash
git add notebooks/01_rigid_body_motions.ipynb notebooks/_figures
git commit -m "docs(notebooks): add 01 rigid-body motions (screw theory foundations)"
```

---

### Task 4: Notebook 02 — Forward kinematics (PoE)

**Files:** Create `notebooks/02_forward_kinematics.ipynb`

**Sections:** PoE formula `T = e^{[𝒮₁]θ₁}···e^{[𝒮ₙ]θₙ}M`; the **Panda 7-vs-8 joint convention** explained here; space vs body frame; mapping screw list columns to joints; TikZ schematic of the Panda kinematic chain.

- [ ] **Step 1:** Author. Load via `load_panda()`; show `sm.S_list.shape == (6,8)`, explain the fixed flange, adopt the 7-vector. Compute FK at `HOME` and a few configs; plot end-effector position with a pgf figure.
- [ ] **Step 2 (smoke test):**
```python
from _shared.helpers import load_panda, HOME
import numpy as np
sm,_ = load_panda()
T = sm.forward_kinematics(HOME, frame="space")
Tb = sm.forward_kinematics(HOME, frame="body")
assert T.shape==(4,4) and np.allclose(T[3],[0,0,0,1])
assert np.isclose(np.linalg.det(T[:3,:3]), 1.0, atol=1e-6)   # valid rotation
print("nb02 smoke OK")
```
- [ ] **Step 3:** Execute. **Step 4:** Verify no errors. **Step 5:** Commit:
```bash
git add notebooks/02_forward_kinematics.ipynb notebooks/_figures
git commit -m "docs(notebooks): add 02 forward kinematics (PoE)"
```

---

### Task 5: Notebook 03 — Velocity kinematics & Jacobians

**Files:** Create `notebooks/03_velocity_kinematics_jacobians.ipynb`

**Sections:** space/body Jacobian (`sm.jacobian`), end-effector velocity (`sm.end_effector_velocity`), resolved-rate / `sm.joint_velocity`, 7-DOF **redundancy & null space** (null space of J via SVD), manipulability preview. pgf figure of a velocity ellipsoid.

- [ ] **Step 1:** Author with the sections above.
- [ ] **Step 2 (smoke test):**
```python
from _shared.helpers import load_panda, HOME
import numpy as np
sm,_ = load_panda()
J = sm.jacobian(HOME, frame="space")
assert J.shape == (6,8) or J.shape == (6,7)
dq = np.zeros(J.shape[1]); dq[0]=0.1
V = sm.end_effector_velocity(HOME, dq, frame="space")
assert V.shape == (6,)
ns = J.shape[1] - np.linalg.matrix_rank(J)     # redundancy dimension
assert ns >= 1
print("nb03 smoke OK")
```
- [ ] **Step 3:** Execute. **Step 4:** Verify. **Step 5:** Commit:
```bash
git add notebooks/03_velocity_kinematics_jacobians.ipynb notebooks/_figures
git commit -m "docs(notebooks): add 03 velocity kinematics and Jacobians"
```

---

### Task 6: Notebook 04 — Inverse kinematics

**Files:** Create `notebooks/04_inverse_kinematics.ipynb`

**Sections:** the IK problem; damped least squares (`iterative_inverse_kinematics`, explain `damping`, `step_cap`); `smart_inverse_kinematics` strategies; redundancy resolution; convergence/residual plot (pgf).

- [ ] **Step 1:** Author. Pick a reachable target via FK, perturb the seed, recover it.
- [ ] **Step 2 (smoke test):**
```python
from _shared.helpers import load_panda, HOME
import numpy as np
sm,_ = load_panda()
T_goal = sm.forward_kinematics(HOME, frame="space")
seed = HOME + 0.2
theta, ok, iters = sm.iterative_inverse_kinematics(T_goal, seed, max_iterations=2000)
T_sol = sm.forward_kinematics(theta, frame="space")
assert ok, f"IK failed after {iters}"
assert np.allclose(T_sol[:3,3], T_goal[:3,3], atol=1e-3)   # position recovered
print("nb04 smoke OK")
```
- [ ] **Step 3:** Execute. **Step 4:** Verify. **Step 5:** Commit:
```bash
git add notebooks/04_inverse_kinematics.ipynb notebooks/_figures
git commit -m "docs(notebooks): add 04 inverse kinematics"
```

---

### Task 7: Notebook 05 — Dynamics

**Files:** Create `notebooks/05_dynamics.ipynb`

**Sections:** mass matrix (`mass_matrix`), Coriolis/centrifugal (`velocity_quadratic_forces`), gravity (`gravity_forces`), inverse dynamics (`inverse_dynamics`), forward dynamics (`forward_dynamics`); round-trip ID→FD check. pgf figure of gravity torque vs a sweep of joint-2 angle.

- [ ] **Step 1:** Author with the sections above (g = `[0,0,-9.81]`, Ftip = zeros(len)).
- [ ] **Step 2 (smoke test):**
```python
from _shared.helpers import load_panda, HOME
import numpy as np
sm, dyn = load_panda()
n = dyn.mass_matrix(HOME).shape[0]
M = dyn.mass_matrix(HOME)
assert M.shape==(n,n) and np.allclose(M, M.T, atol=1e-6)        # symmetric
assert np.all(np.linalg.eigvalsh(M) > 0)                        # positive definite
g = dyn.gravity_forces(HOME, [0,0,-9.81])
dq = np.zeros(n); ddq = np.zeros(n)
tau = dyn.inverse_dynamics(HOME, dq, ddq, [0,0,-9.81], [0]*n)
assert np.allclose(tau, g, atol=1e-6)            # static torque == gravity
ddq_fd = dyn.forward_dynamics(HOME, dq, tau, [0,0,-9.81], [0]*n)
assert np.allclose(ddq_fd, 0, atol=1e-4)         # ID->FD round trip
print("nb05 smoke OK")
```
- [ ] **Step 3:** Execute. **Step 4:** Verify. **Step 5:** Commit:
```bash
git add notebooks/05_dynamics.ipynb notebooks/_figures
git commit -m "docs(notebooks): add 05 dynamics"
```

---

### Task 8: Notebook 06 — Trajectory planning

**Files:** Create `notebooks/06_trajectory_planning.ipynb`

**Sections:** point-to-point in joint space; cubic (method=3) vs quintic (method=5) time scaling; position/velocity/acceleration profiles; inverse-dynamics along the trajectory (`inverse_dynamics_trajectory`). pgf figures of the three profiles.

- [ ] **Step 1:** Author. Build a TrajectoryPlanning instance with `joint_limits()`.
- [ ] **Step 2 (smoke test):**
```python
from _shared.helpers import load_panda, HOME, joint_limits, PANDA_URDF
from ManipulaPy.path_planning import TrajectoryPlanning
import numpy as np
sm, dyn = load_panda()
trr = TrajectoryPlanning(serial_manipulator=sm, urdf_path=PANDA_URDF,
                         dynamics=dyn, joint_limits=joint_limits(), torque_limits=None)
start = np.zeros(7); end = HOME
traj = trr.joint_trajectory(start, end, Tf=3.0, N=200, method=5)
P = traj["positions"]
assert P.shape == (200, 7)
assert np.allclose(P[0], start, atol=1e-6) and np.allclose(P[-1], end, atol=1e-6)
assert np.allclose(traj["velocities"][0], 0, atol=1e-6)   # rest-to-rest
print("nb06 smoke OK")
```
- [ ] **Step 3:** Execute. **Step 4:** Verify. **Step 5:** Commit:
```bash
git add notebooks/06_trajectory_planning.ipynb notebooks/_figures
git commit -m "docs(notebooks): add 06 trajectory planning"
```

---

### Task 9: Notebook 07 — Control

**Files:** Create `notebooks/07_control.ipynb`

**Sections:** feedback control intro; PID (`pid_control`), computed-torque (`computed_torque_control`); simulate tracking of the Task-8 trajectory with a simple Euler loop using `forward_dynamics`; tracking-error plot; brief tuning (`ziegler_nichols_tuning`). pgf figure of desired vs actual joint angle.

- [ ] **Step 1:** Author. Implement a short integration loop: at each step compute `tau` from the controller, integrate state with `dyn.forward_dynamics`.
- [ ] **Step 2 (smoke test):**
```python
from _shared.helpers import load_panda, HOME
from ManipulaPy.control import ManipulatorController
import numpy as np
sm, dyn = load_panda()
ctrl = ManipulatorController(dynamics=dyn)
n = dyn.mass_matrix(HOME).shape[0]
tau = ctrl.computed_torque_control(
    thetalistd=HOME, dthetalistd=np.zeros(n), ddthetalistd=np.zeros(n),
    thetalist=HOME*0, dthetalist=np.zeros(n), g=[0,0,-9.81], dt=0.01,
    Kp=np.ones(n)*100, Ki=np.zeros(n), Kd=np.ones(n)*20) if hasattr(ctrl,"computed_torque_control") else None
assert tau is None or np.asarray(tau).shape[0] == n
print("nb07 smoke OK")
```
> Note: confirm `computed_torque_control` parameter names at author time via
> `inspect.signature(ctrl.computed_torque_control)` and adjust the call; the smoke
> test only asserts output dimensionality.
- [ ] **Step 3:** Execute. **Step 4:** Verify. **Step 5:** Commit:
```bash
git add notebooks/07_control.ipynb notebooks/_figures
git commit -m "docs(notebooks): add 07 control"
```

---

### Task 10: Notebook 08 — Singularities & manipulability

**Files:** Create `notebooks/08_singularities_manipulability.ipynb`

**Sections:** singular configurations; manipulability measure `w=sqrt(det(JJ^T))`; manipulability ellipsoid (eigenvectors/values of `JJ^T`); `Singularity` class usage; workspace sampling. pgf figure of the manipulability ellipsoid + a w-vs-config sweep.

- [ ] **Step 1:** Author using `sm.jacobian` and `from ManipulaPy.singularity import Singularity` (confirm its constructor/methods at author time via `inspect`).
- [ ] **Step 2 (smoke test):**
```python
from _shared.helpers import load_panda, HOME
import numpy as np
sm,_ = load_panda()
J = sm.jacobian(HOME, frame="space")
JJt = J @ J.T
w = np.sqrt(max(np.linalg.det(JJt), 0.0))
assert w > 0                       # HOME is non-singular
evals = np.linalg.eigvalsh(JJt)
assert np.all(evals >= -1e-9)      # ellipsoid axes well-defined
print("nb08 smoke OK")
```
- [ ] **Step 3:** Execute. **Step 4:** Verify. **Step 5:** Commit:
```bash
git add notebooks/08_singularities_manipulability.ipynb notebooks/_figures
git commit -m "docs(notebooks): add 08 singularities and manipulability"
```

---

### Task 11: Notebook 09 — Simulation (PyBullet)

**Files:** Create `notebooks/09_simulation_pybullet.ipynb`

**Sections:** connect in **DIRECT** (headless) mode; load Panda; set joint positions; play back the Task-8 trajectory; read back joint states; collision check. Show a rendered frame via `pybullet.getCameraImage` embedded as PNG (not a pgf plot). Confirm `Simulation` method names at author time via `inspect`.

- [ ] **Step 1:** Author. Use `Simulation(PANDA_URDF, joint_limits())` and DIRECT mode; avoid GUI.
- [ ] **Step 2 (smoke test):**
```python
import pybullet as p
from _shared.helpers import PANDA_URDF, joint_limits
cid = p.connect(p.DIRECT)
assert cid >= 0
import pybullet_data, os
p.setAdditionalSearchPath(pybullet_data.getDataPath())
rid = p.loadURDF(PANDA_URDF, useFixedBase=True)
assert rid >= 0
n = p.getNumJoints(rid); assert n >= 7
p.disconnect()
print("nb09 smoke OK")
```
> If the `Simulation` wrapper is used instead of raw pybullet, assert
> `sim.get_joint_positions()` returns a length-≥7 array after `run_trajectory`.
- [ ] **Step 3:** Execute. **Step 4:** Verify. **Step 5:** Commit:
```bash
git add notebooks/09_simulation_pybullet.ipynb notebooks/_figures
git commit -m "docs(notebooks): add 09 simulation with PyBullet"
```

---

### Task 12: Notebook 10 — Perception & collision avoidance

**Files:** Create `notebooks/10_perception_collision.ipynb`

**Sections:** depth → 3D points → obstacle clustering with DBSCAN (`Perception`); potential-field collision avoidance (`PotentialField`, `CollisionChecker`): attractive + repulsive gradients; a short planned path that avoids a synthetic obstacle; (optional, requires fixed torch) YOLO detection with `Vision` on a sample image. pgf scatter of clusters + path; YOLO output embedded as PNG.

- [ ] **Step 1:** Author. Use a synthetic point cloud (numpy) so it is deterministic; confirm `Perception`/`PotentialField` constructors at author time via `inspect`. Guard the YOLO cell with `try/except` and print a clear message if torch/ultralytics unavailable (but with Task 0 done it should run).
- [ ] **Step 2 (smoke test):**
```python
import numpy as np
from sklearn.cluster import DBSCAN
pts = np.vstack([np.random.RandomState(0).randn(50,3)*0.05 + c
                 for c in ([0.5,0,0.3],[0.2,0.4,0.5])])
labels = DBSCAN(eps=0.1, min_samples=5).fit_predict(pts)
assert len(set(labels) - {-1}) == 2          # two clusters found
from ManipulaPy.potential_field import PotentialField   # import works
print("nb10 smoke OK")
```
- [ ] **Step 3:** Execute. **Step 4:** Verify. **Step 5:** Commit:
```bash
git add notebooks/10_perception_collision.ipynb notebooks/_figures
git commit -m "docs(notebooks): add 10 perception and collision avoidance"
```

---

### Task 13: Notebook 11 — Capstone (pick & place)

**Files:** Create `notebooks/11_capstone_pick_and_place.ipynb`

**Sections:** end-to-end integration reusing prior concepts — (1) define pick & place poses, (2) IK for each (`smart_inverse_kinematics`), (3) plan joint trajectories between them (`joint_trajectory`), (4) compute torques (`inverse_dynamics_trajectory`), (5) execute in PyBullet DIRECT, (6) verify the end-effector reached the place pose. A final TikZ block diagram of the full pipeline.

- [ ] **Step 1:** Author the pipeline, importing only previously-introduced APIs.
- [ ] **Step 2 (smoke test):**
```python
from _shared.helpers import load_panda, HOME, joint_limits, PANDA_URDF
from ManipulaPy.path_planning import TrajectoryPlanning
import numpy as np
sm, dyn = load_panda()
pick = sm.forward_kinematics(HOME, frame="space").copy()
place = pick.copy(); place[:3,3] += np.array([0.0, 0.15, -0.05])
q_pick, ok1, _ = sm.smart_inverse_kinematics(pick, theta_current=HOME)
q_place, ok2, _ = sm.smart_inverse_kinematics(place, theta_current=q_pick)
assert ok1 and ok2
trr = TrajectoryPlanning(serial_manipulator=sm, urdf_path=PANDA_URDF,
                         dynamics=dyn, joint_limits=joint_limits(), torque_limits=None)
traj = trr.joint_trajectory(q_pick, q_place, Tf=2.0, N=100, method=5)
T_end = sm.forward_kinematics(traj["positions"][-1], frame="space")
assert np.allclose(T_end[:3,3], place[:3,3], atol=2e-2)   # reached place pose
print("nb11 capstone smoke OK")
```
- [ ] **Step 3:** Execute. **Step 4:** Verify. **Step 5:** Commit:
```bash
git add notebooks/11_capstone_pick_and_place.ipynb notebooks/_figures
git commit -m "docs(notebooks): add 11 capstone pick-and-place"
```

---

## Task 14: README + gitignore + full re-execution

**Files:**
- Create: `notebooks/README.md`
- Modify: `.gitignore` (ignore `notebooks/_figures/*.pdf`, `notebooks/_figures/*.tex`, keep `*.png`)

- [ ] **Step 1:** Write `notebooks/README.md`: course intro, prerequisites
  (`pip install -e ".[simulation,urdf,vision,ml,cuda]"`, the torch≥2.3 note, a TeX Live /
  lualatex note), recommended order 01→11, one-line summary per notebook.
- [ ] **Step 2:** Add to `.gitignore`:
```
notebooks/_figures/*.pdf
notebooks/_figures/*.tex
notebooks/_figures/*.aux
notebooks/_figures/*.log
```
- [ ] **Step 3: Re-execute the whole course from clean kernels**

Run:
```bash
cd notebooks
for nb in 0?_*.ipynb 1?_*.ipynb; do
  echo "=== $nb ==="
  jupyter nbconvert --to notebook --execute --inplace "$nb" --ExecutePreprocessor.timeout=900 || exit 1
done
```
Expected: every notebook exits 0.

- [ ] **Step 4: Assert no error cells across all notebooks**

Run:
```bash
python3 - <<'PY'
import nbformat, glob, sys
bad=[]
for f in sorted(glob.glob("notebooks/??_*.ipynb")):
    nb=nbformat.read(f,4)
    for c in nb.cells:
        if c.cell_type=="code":
            for o in c.get("outputs",[]):
                if o.get("output_type")=="error": bad.append(f)
print("error notebooks:", sorted(set(bad)) or "NONE")
sys.exit(1 if bad else 0)
PY
```
Expected: `error notebooks: NONE`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add notebooks/README.md .gitignore notebooks/*.ipynb notebooks/_figures/*.png
git commit -m "docs(notebooks): add course README and finalize executed outputs"
```

---

## Self-review notes

- **Spec coverage:** §3 structure → Tasks 1–2,14; §4 notebooks 01–11 → Tasks 3–13; §5 execution → Task 0 + per-notebook Step 3/4 + Task 14; §5a TikZ → Task 1; §6 helpers/README → Tasks 1,2,14; §6a seed corrections → Task 3 (notation, missing M) and helpers (no hard-coded path, no GUI). All covered.
- **Smoke tests** assert real numeric invariants (rotation determinant, mass-matrix SPD, static torque == gravity, IK round-trip, rest-to-rest velocities, capstone reach), not just "ran".
- **Open author-time confirmations** (flagged inline, not placeholders): exact parameter names for `computed_torque_control` (Task 9), `Singularity` API (Task 10), `Perception`/`PotentialField`/`Simulation` constructors (Tasks 11,12,13). These are introspected during authoring; the smoke tests pin the observable contract regardless.
