# ManipulaPy

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/ManipulaPy)](https://pypi.org/project/ManipulaPy/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
![CI](https://github.com/boelnasr/ManipulaPy/actions/workflows/test.yml/badge.svg?branch=main)
[![codecov](https://codecov.io/gh/boelnasr/ManipulaPy/branch/main/graph/badge.svg)](https://codecov.io/gh/boelnasr/ManipulaPy)
[![status](https://joss.theoj.org/papers/e0e68c2dcd8ac9dfc1354c7ee37eb7aa/status.svg)](https://joss.theoj.org/papers/e0e68c2dcd8ac9dfc1354c7ee37eb7aa)

**A modern, GPU-accelerated Python package for robot manipulator kinematics, dynamics, planning, simulation, control, and perception.**

[Quick start](#quick-start) • [Documentation](https://manipulapy.readthedocs.io/) • [Examples](Examples/) • [Changelog](CHANGELOG.md) • [Contributing](CONTRIBUTING.md)

<img src="docs/source/_static/gifs/ur5_pick_motion.gif" alt="UR5 executing a quintic-timed reach trajectory in PyBullet" width="540">

</div>

---

## Why ManipulaPy

Most Python robotics packages cover one slice well — kinematics, simulation, or perception — and force you to glue the rest together. ManipulaPy ships the full stack with a consistent API:

- **Unified surface** — kinematics, dynamics, control, planning, simulation, and vision share the same `SerialManipulator` / `ManipulatorDynamics` objects.
- **GPU when it pays, CPU when it doesn't** — CUDA trajectory and dynamics kernels auto-switch on problem size; the default install is lightweight (NumPy/SciPy/Matplotlib/Numba/Pillow) and heavy deps live behind optional extras.
- **Production-ready URDF** — native NumPy 2.0–compatible parser with `package://`, `file://`, and ROS package discovery built in.
- **25 bundled robots** — Universal Robots, Fanuc, KUKA, Kinova, Franka, UFactory, Robotiq, ABB. Load any of them by name.

---

## Quick start

### Install

```bash
# Lightweight default — kinematics, dynamics, control, native URDF parser, CPU trajectories
pip install ManipulaPy

# Add the features you need:
pip install "ManipulaPy[simulation]"   # PyBullet physics + visualization
pip install "ManipulaPy[urdf]"         # trimesh-backed mesh loading
pip install "ManipulaPy[vision]"       # OpenCV + Ultralytics YOLO + torch
pip install "ManipulaPy[ml]"           # scikit-learn (DBSCAN clustering)
pip install "ManipulaPy[cuda]"         # CuPy 12.x for CUDA 12.x toolchains
pip install "ManipulaPy[all]"          # everything above
```

For CUDA 11.x toolchains use `[gpu-cuda11]`; for AMD/ROCm use `[gpu-rocm]`. Full matrix in the [Installation Guide](docs/source/Installation%20Guide.rst).

### Verify

```python
import ManipulaPy
ManipulaPy.check_dependencies()    # ✅/❌ for each feature
```

### 30-second demo

```python
import numpy as np
from ManipulaPy.urdf import URDF
from ManipulaPy.ManipulaPy_data import get_robot_urdf
from ManipulaPy.path_planning import OptimizedTrajectoryPlanning

# Load any of the 25 bundled robots by name
robot_urdf = get_robot_urdf("ur5")
robot = URDF.load(robot_urdf)
serial = robot.to_serial_manipulator()
dynamics = robot.to_manipulator_dynamics()

# Forward kinematics
joint_angles = np.array([0.1, 0.2, -0.3, -0.5, 0.2, 0.1])
T = serial.forward_kinematics(joint_angles)
print("end-effector:", T[:3, 3])

# Trajectory planning — auto-switches to GPU when problem is large enough
planner = OptimizedTrajectoryPlanning(
    serial, robot_urdf, dynamics, joint_limits=[(-np.pi, np.pi)] * 6,
)
traj = planner.joint_trajectory(
    thetastart=np.zeros(6), thetaend=joint_angles,
    Tf=5.0, N=1000, method=5,   # 5 = quintic; 3 = cubic; 1 = linear
)
print(f"trajectory: {traj['positions'].shape[0]} points")
```

That snippet runs end-to-end on a default `pip install ManipulaPy` — no GPU required. With the `[cuda]` extra installed, the planner transparently routes large problems (≥ ~1000 waypoints) through the CUDA kernels for 40×+ speedup.

### What it looks like

<table>
<tr>
<td width="33%" align="center">

<img src="docs/source/_static/gifs/joint_trajectory.gif" alt="Six joint angles unrolling through a quintic-timed trajectory" width="100%">

**Trajectory planning**

Quintic time-scaled joint trajectory, CPU or CUDA.

</td>
<td width="33%" align="center">

<img src="docs/source/_static/gifs/ee_path.gif" alt="UR5 end-effector tracing the same path in 3D" width="100%">

**Forward kinematics**

End-effector path computed from the same trajectory.

</td>
<td width="33%" align="center">

<img src="docs/source/_static/gifs/workspace.gif" alt="Monte-Carlo reachable workspace of the UR5" width="100%">

**Workspace analysis**

Monte-Carlo reachable workspace, GPU-accelerated.

</td>
</tr>
</table>

Both the abstract trajectory plots (above) and the PyBullet-rendered robot clips (hero + robot zoo) are regenerated from the live API by [`scripts/record_readme_gifs.py`](scripts/record_readme_gifs.py) and [`scripts/record_robot_gifs.py`](scripts/record_robot_gifs.py) — re-run after a release to keep the visuals in sync.

---

## Features

### Core (always available — pure NumPy/SciPy/Numba)

- **Kinematics** — forward + inverse (DLS, SQP, TRAC-IK, multi-start), Jacobians, geometric error model
- **Dynamics** — mass matrix, Coriolis/centrifugal, gravity, inverse/forward dynamics
- **Control** — PID, computed torque, adaptive, robust, Kalman filtering, Ziegler-Nichols auto-tuning
- **Singularity analysis** — manipulability ellipsoid, condition number, Monte-Carlo workspace
- **Native URDF parser** — `package://`, `file://`, ROS package discovery, explicit `PackageResolver` overrides

### With optional extras

- **`[simulation]`** — PyBullet physics, GUI sliders, collision checking, trajectory replay
- **`[urdf]`** — trimesh-backed mesh loading for visualization
- **`[vision]`** — OpenCV + Ultralytics YOLO + stereo + 3D point clouds
- **`[ml]`** — DBSCAN-based obstacle clustering on top of vision
- **`[cuda]`** — CuPy/Numba CUDA kernels: trajectory generation (40×+), batch trajectories (20×+), inverse dynamics (100×+), Monte-Carlo workspace (10×+)

### Bundled robots

UR3 / UR5 / UR10 / UR3e / UR5e / UR10e / UR16e · Fanuc LR Mate 200iB, M-16iB, CRX-5/10/20/30iA · KUKA iiwa7 / iiwa14 · Kinova Gen3, Jaco 6-DOF, Jaco 7-DOF · Franka Panda · UFactory xArm6 (± gripper) · Robotiq 2F-85 / 2F-140 · ABB IRB 2400.

<p align="center">
  <img src="docs/source/_static/gifs/robot_zoo.gif" alt="UR5, Panda, iiwa14, and xArm6 in PyBullet — bundled URDFs rendered via ManipulaPy.urdf.PackageResolver" width="540">
</p>

Every model loads end-to-end through `ManipulaPy.urdf.URDF.load(...)` and renders in PyBullet via `ManipulaPy.urdf.PackageResolver` (the GIF above is generated by [`scripts/record_robot_gifs.py`](scripts/record_robot_gifs.py) — no ROS or external mesh setup required).

```python
from ManipulaPy.ManipulaPy_data import list_robots, print_robot_catalog
print(list_robots())          # iterable of robot keys
print_robot_catalog()         # printable table with specs
```

Full inventory and per-robot details in [`ManipulaPy/ManipulaPy_data/MANIFEST.md`](ManipulaPy/ManipulaPy_data/MANIFEST.md).

---

## Documentation

| | |
|---|---|
| **Tutorials & user guide** | [manipulapy.readthedocs.io](https://manipulapy.readthedocs.io/) |
| **API reference** | [API docs](https://manipulapy.readthedocs.io/en/latest/api/index.html) |
| **Installation matrix** | [`docs/source/Installation Guide.rst`](docs/source/Installation%20Guide.rst) |
| **Runnable examples** | [`Examples/`](Examples/) — basic, intermediate, advanced tracks |
| **Architecture** | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| **Release history** | [`CHANGELOG.md`](CHANGELOG.md) |

The `Examples/` tree is the fastest way in. Start at `Examples/basic_examples/` (no extras required), move to `Examples/intermediate_examples/`, then `Examples/advanced_examples/` for the full GPU + vision pipelines.

---

## What's new in v1.3.2

The full release notes are in [CHANGELOG.md](CHANGELOG.md). Highlights:

- **Modular optional extras** — lightweight default install; heavy deps (PyBullet, trimesh, OpenCV, ultralytics, torch, sklearn, CuPy) opt in via `[simulation]`, `[urdf]`, `[vision]`, `[ml]`, `[cuda]`, `[all]`.
- **Native URDF parser** — `ManipulaPy.urdf.URDF` + `PackageResolver`. NumPy 2.0 compatible, no urchin dependency.
- **CUDA kernel correctness** — corrected quintic acceleration; removed shared-memory and forward-dynamics races; added `method=1` (linear) to every kernel variant; N ≤ 1 div-zero guards; **fixed the repulsive-potential gradient sign** (previous versions silently attracted the robot toward obstacles).
- **Simulation guards** — every `Simulation` method that touches PyBullet now raises a clear `ImportError("pip install ManipulaPy[simulation]")` when the extra is missing.
- **Vision defaults** — `Vision.detect_obstacles(depth_threshold=5.0)` (was 0.0, which silently filtered everything).
- **Kalman filter validation** — `kalman_filter_update` validates both `x_hat` and `P` shape before matrix algebra; `calculate_settling_time` returns the first settled time and handles negative setpoints.
- **PEP 561** — `py.typed` marker ships in the wheel; mypy/pyright honor the in-source type hints.
- **Python 3.12** in the CI matrix and PyPI classifiers.

---

## Performance

GPU vs CPU on the bundled `Benchmark/` suite, RTX 3060 + Ryzen 7, NumPy 2.2 + CuPy 13:

| Workload | Problem size | Speedup |
|---|---|---|
| Joint trajectory (quintic) | N = 5,000 × 6 joints | ~45× |
| Batch trajectory generation | 64 trajectories × N = 1,000 | ~22× |
| Inverse dynamics over trajectory | N = 10,000 | ~110× |
| Monte-Carlo workspace | 50,000 samples | ~12× |

The planner auto-routes to CPU below the `cuda_threshold` (default 200 waypoints) so small problems don't pay PCIe transfer overhead. Reproduce locally with `python -m ManipulaPy.Benchmark.quick_benchmark`.

---

## Contributing

Bug reports, feature requests, and pull requests welcome. The flow is documented in [`CONTRIBUTING.md`](CONTRIBUTING.md); the short version:

1. Fork → branch → make the change → `python -m pytest tests/ -q` should be green.
2. New behavior needs a regression test in `tests/test_v132_regressions.py` (or a sibling file).
3. Surgical edits over speculative refactors — see CLAUDE.md if you're collaborating with an AI assistant.
4. Open a PR against `main`. CI runs Python 3.8 – 3.12.

---

## Citation

If you use ManipulaPy in academic work, please cite:

```bibtex
@software{manipulapy2026,
  title   = {ManipulaPy: A Comprehensive Python Package for Robotic Manipulator Analysis and Control},
  author  = {Mohamed Aboelnasr},
  year    = {2026},
  url     = {https://github.com/boelnasr/ManipulaPy},
  version = {1.3.2},
  license = {AGPL-3.0-or-later}
}
```

A JOSS paper is in review — [submission status](https://joss.theoj.org/papers/e0e68c2dcd8ac9dfc1354c7ee37eb7aa).

---

## License

[AGPL-3.0-or-later](LICENSE.md). Free for research, education, and AGPL-compatible commercial use; network-deployed services must publish source.

All runtime dependencies are AGPL-compatible: NumPy/SciPy/Matplotlib (BSD), Numba/CuPy (BSD/MIT), Pillow (HPND), PyBullet (Zlib), OpenCV (Apache 2.0), Ultralytics (AGPL-3.0), Trimesh (MIT).

---

## Support

- 📚 [Documentation](https://manipulapy.readthedocs.io/)
- 🐛 [Issues](https://github.com/boelnasr/ManipulaPy/issues)
- 💬 [Discussions](https://github.com/boelnasr/ManipulaPy/discussions)
- 📧 [aboelnasr1997@gmail.com](mailto:aboelnasr1997@gmail.com)

Maintained by [Mohamed Aboelnasr](https://github.com/boelnasr).
