# Architecture

This document describes the internal architecture of ManipulaPy: class hierarchy,
data flow, lazy loading, GPU/CPU strategy, and the URDF parser backends.

## Class Hierarchy & Data Flow

```
URDF File → URDFToSerialManipulator (factory, composition)
                    │
                    ├→ SerialManipulator (base: FK/IK, Jacobian)
                    │         │
                    └→ ManipulatorDynamics(SerialManipulator) (adds mass matrix, Coriolis, gravity)
                                  │
                                  └→ ManipulatorController (composition, wraps dynamics)
                                         └→ 14+ control algorithms (PID, computed torque, adaptive, etc.)
```

- `SerialManipulator` is the core class. Uses product-of-exponentials with screw axes (S_list/B_list) and home config (M_list).
- `ManipulatorDynamics` **inherits** from `SerialManipulator`, adding Glist (inertia matrices) and cached mass matrix / Coriolis computations.
- `ManipulatorController` uses **composition** (wraps a `ManipulatorDynamics` instance).
- `URDFToSerialManipulator` is a **factory** — it parses URDF and creates `SerialManipulator` or `ManipulatorDynamics` instances.

## Lazy Loading System

`__init__.py` uses `__getattr__` to load modules on-demand. `import ManipulaPy` is ~50ms; heavy modules (vision, sim, cuda_kernels) only load when accessed. The `_module_cache` dict prevents re-imports. Feature availability is detected lazily via `importlib.util.find_spec()` without actually importing dependencies.

## GPU/CPU Fallback Strategy

Three tiers, transparent to calling code:
1. **GPU (CuPy/CUDA)**: Used for batch operations (N>1000) in `path_planning.py` and `cuda_kernels.py`
2. **CPU Optimized (Numba JIT)**: Used for medium workloads
3. **Pure NumPy**: Always-available fallback

The **control module is CPU-only by design** — `_to_numpy()` (defined in `control.py`) converts all CuPy arrays to NumPy before computation because GPU↔CPU PCIe transfer latency exceeds the compute savings for single-sample real-time control loops.

## URDF Parser Backends

`urdf_processor.py` supports two backends (set via `parser_backend` parameter):
- `"builtin"` (default): Native parser in `ManipulaPy/urdf/`, NumPy 2.0+ compatible, zero external deps
- `"pybullet"`: Uses PyBullet for parsing

The `"urchin"` backend was removed in v1.3.0; no urchin import remains in the codebase.

## Optional Dependency Contract

`ManipulaPy.sim` and `ManipulaPy.control` must remain importable even when
`cupy` and/or `pybullet` are absent. Neither module raises `ImportError` at
module load time; the error is deferred to the moment a feature that actually
needs the missing dependency is called.

- Methods that require `pybullet` call the module-level `_check_pybullet_available()`
  helper at entry, before touching any `p.*` symbol. The helper raises
  `ImportError` with a clear install hint (`pip install 'ManipulaPy[simulation]'`).
- `cupy` absence is handled by a `_NumpyProxy` wrapper that delegates attribute
  access to NumPy, so import-time code that inspects `cp.*` works without a GPU.

**Contributor rule:** every new method added to `ManipulaPy.sim` that calls into
PyBullet must begin with `_check_pybullet_available()`, AND must have an
accompanying regression test in
`tests/test_v132_regressions.py::TestSimPybulletGuards` that verifies the guard
fires before any PyBullet symbol is reached.

## Self-Collision

`Simulation.__init__` accepts two opt-in parameters (both default to disabled,
preserving v1.3.1 behavior):

- `enable_self_collision: bool = False` — when `True`, passes PyBullet's
  `URDF_USE_SELF_COLLISION` flag to `loadURDF`.
- `disable_pairs: list[tuple[str, str]] | None = None` — per-link-pair
  exclusions applied via `setCollisionFilterPair` after the robot is loaded.

`Simulation.check_collisions()` returns `list[tuple[int, int, tuple]]` —
each element is `(linkA, linkB, position)` from PyBullet contact points
queried without a per-joint filter so base-link (index −1) contacts are
included.

`potential_field.CollisionChecker` builds an Allowed Collision Matrix (ACM)
from URDF joint topology (parent↔child and grandparent↔grandchild pairs) via
`build_link_adjacency`, suppressing adjacent-link false positives. Mesh geometry
is sourced from `link.collisions` in preference to `link.visuals`.

Note: the simplified SAT in `_points_intersect` is intentionally conservative
for v1.3.2. FCL-backed exact collision is scheduled for v1.5.0.

## Code Conventions

- **Formatting**: black + isort, both line-length 88, isort profile "black". All configs in `pyproject.toml`.
- **Linting**: flake8 (config in `.flake8`, ignores F403/F405 for wildcard imports in `__init__.py`)
- **Tests**: pytest with `unittest.TestCase` style (not pure pytest functions). Google-style docstrings.
- **Version**: Must be updated in sync in `pyproject.toml` and `ManipulaPy/__init__.py`

## Test Infrastructure

`tests/conftest.py` implements a smart mocking system:
- **Unconditionally mocked**: `pycuda`, `pycuda.driver`, `pycuda.autoinit`, `torchvision` (plus `torchvision.ops`/`transforms`/`io`)
- **Mocked when `FORCE_CPU_ONLY` is active** (no GPU detected via `/dev/nvidiactl` + `nvidia-smi`, or `MANIPULAPY_FORCE_CPU=1` set): `cupy`, `numba.cuda`, `numba.cuda.random`
- **Mocked in CI** (no GPU hardware): `pybullet`
- **Tested when available**: `torch`, `cv2`, `sklearn`, `ultralytics`, `numba`

`FORCE_CPU_ONLY` is determined at conftest load time: if `/dev/nvidiactl` or `/dev/nvidia[0-9]*` devices are absent and `nvidia-smi` is not on `$PATH`, the flag is set and `MANIPULAPY_FORCE_CPU=1` is written to the environment. Users can also set `MANIPULAPY_FORCE_CPU=1` explicitly to force CPU mode regardless of hardware. On NVIDIA hosts the real CuPy / Numba CUDA paths are allowed to load and decide availability for themselves. Never import cupy/pycuda directly in tests — conftest handles it.

### Module-to-Test Mapping

- kinematics → test_kinematics.py, test_robust_ik_demo.py
- dynamics → test_dynamics.py
- control → test_control.py, test_control_unit.py
- path_planning → test_path_planning_cpu.py, test_path_planning_unit.py, test_trajectory_planning.py
- potential_field → test_potential_field.py, test_potential_field_extended.py
- singularity → test_singularity.py, test_singularity_extended.py
- urdf_processor → test_urdf_processor.py, test_urdf_native.py, test_urdf_accuracy.py, test_urdf_comparison.py
- sim → test_sim.py | vision → test_vision.py | perception → test_perception.py
- cuda_kernels → test_cuda_kernels.py, test_cuda_kernels_cpu.py | utils → test_utils.py

## Gotchas

- `ManipulaPy_data/` contains 25 robot URDFs (~143MB with meshes) — exclude from searches
- `.venv/` exists at project root — exclude from searches
- Wildcard imports in `__init__.py` are intentional (F403/F405 ignored in flake8/ruff)
- mypy overrides ignore all third-party robotics/ML packages (pybullet, cupy, torch, etc.)
- URDF test fixtures depend on `ManipulaPy_data` package data being installed
