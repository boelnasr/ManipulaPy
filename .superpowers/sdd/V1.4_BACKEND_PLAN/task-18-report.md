# Task 18 Report: GPU kernel registry + additional kernels

Status: **DONE_WITH_CONCERNS**

## Outcome

- Added a deterministic `KernelRegistry` and immutable `KernelRegistration` model carrying the operation name, raw Numba implementation, launch configuration, explicit CPU fallback, backend-specific launchers, and immutable metadata.
- Registered trajectory generation variants and the fused potential-field operation without adding top-level `ManipulaPy.cuda_kernels` exports.
- Routed the existing trajectory and potential-field wrappers through the Task 9 active-backend GPU-capability predicate. GPU-capable backends select the existing CUDA launchers; other backends select explicit NumPy fallbacks.
- Duplicate registration raises `ValueError`; unknown lookup raises a deterministic `KeyError` listing sorted available names.
- Added CPU fallback parity/hand-checked tests and hard-assertion CUDA tests marked for the Task 17 GPU lane.

## RED / GREEN evidence

The implementation and tests arrived as an uncommitted finish-only handoff, so a pre-implementation test RED was not recreated or fabricated. The first bounded verification command was RED at pytest startup because an unrelated auto-loaded ROS `launch_testing` plugin imports the unavailable `lark` package; no project tests collected. Third-party plugin autoload was then disabled and the repository-required timeout plugin loaded explicitly.

GREEN:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NUMBA_DISABLE_CUDA=1 MPLBACKEND=Agg \
  .venv/bin/python -m pytest -p pytest_timeout \
  tests/test_cuda_kernels.py tests/test_cuda_kernels_cpu.py \
  tests/test_trajectory_planning.py -q

80 passed, 35 skipped, 1 warning in 9.73s
```

CUDA-marked tests were skipped because CUDA was deliberately disabled for bounded local verification.

Public API freeze:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NUMBA_DISABLE_CUDA=1 MPLBACKEND=Agg \
  .venv/bin/python -m pytest -p pytest_timeout \
  tests/test_public_api_freeze.py -q

1 passed, 1 warning in 2.84s
```

Formatting/lint:

```text
.venv/bin/python -m black --check <owned Python paths>
.venv/bin/python -m isort --check-only --diff <owned Python paths>
.venv/bin/python -m flake8 <owned Python paths>
git diff --check

All exited 0 with no findings.
```

`ruff` was not installed in the project virtual environment, so the configured Black, isort, and flake8 checks were used.

## Raw-body and export evidence

- The Task 18 diff changes only wrapper/fallback code after the CUDA raw bodies in `trajectory_kernels.py` and `field_kernels.py`; no `@cuda.jit` body is modified.
- Existing raw-kernel AST hash assertions, including all trajectory variants and `fused_potential_gradient_kernel`, passed in `tests/test_cuda_kernels_cpu.py`.
- `ManipulaPy/cuda_kernels/__init__.py` is unchanged by Task 18.
- `tests/test_public_api_freeze.py` passed, proving the frozen public return/export contract remains unchanged.

## Design rationale and self-review

- The registry is operation-oriented rather than exporting new raw kernels. This keeps launch policy and CPU fallback metadata addressable while preserving the compatibility facade.
- Registry entries are frozen dataclasses and metadata is wrapped in `MappingProxyType`, preventing consumers from silently mutating shared contracts.
- Registration is fail-closed: duplicates cannot replace existing operations, and unknown names do not fall through to an arbitrary default.
- Dispatch uses the central `_cuda_routing_enabled()` predicate at execution time, so changing the active backend changes routing without rebuilding the registry.
- GPU launchers call the existing transfer/launch implementations; CPU launchers discard transfer-only options and call explicit numerical references.
- The potential-field fallback mirrors the existing fused CUDA equations, float32 outputs, obstacle influence boundary, and zero-distance exclusion. No new robotics equations were invented.
- Unknown legacy trajectory `kernel_type` values retain prior behavior by resolving to the standard variant.
- Review found no concrete correctness, formatting, ownership, or API-freeze defect requiring an additional patch beyond the inherited implementation.

## Mandatory concern

Live CUDA was not executed locally. The two new CUDA registry tests are hard assertions (not exception-swallowing smoke tests) and are marked `cuda`, but Task 18 is not fully GPU-verified until Task 17's self-hosted runner executes them with a GPU-capable active backend and CuPy installed.
