# Handoff: unbounded `numba` pin let NumPy 2.5.2 into CI/installs unreviewed

Status: **investigated, not yet fixed**. No code changes have been made for
this issue — this doc is the handoff so it isn't lost. Discovered as a
side-effect while working PR #83 (Codecov → `python-coverage-comment-action`
swap); unrelated to that PR's actual changes.

## TL;DR

`pyproject.toml` pins `numba>=0.60` with **no upper bound**, and
`numpy>=2.0,<3.0`. Because numba itself pins NumPy (currently
`numpy<2.6,>=1.22` as of numba 0.67.0), ManipulaPy's *effective* NumPy
ceiling is silently controlled by whatever numba ships next — not by
anything in this repo. When numba 0.67.0 was published on PyPI
(**2026-08-11**), it raised its own ceiling from `numpy<2.5` to `numpy<2.6`,
which let pip's resolver jump straight to NumPy 2.5.2 for any fresh
`pip install manipulapy` from that moment on — no ManipulaPy release, no
review, no changelog entry.

This is happening **right now**, for real installs, not just CI.

## What broke, concretely

Three `backend-cpu` matrix jobs and the Python-3.12 leg of
`non-tpu-regression` in `.github/workflows/test.yml` failed against
numba 0.67.0 + numpy 2.5.2. Full non-tpu-regression run under that combo
(job: `non-tpu-regression (3.12)`, PR #83, run 31692943775):

```
4 failed, 1109 passed, 453 skipped, 6 deselected in 176.83s
```

All 4 failures are golden-snapshot / API-freeze meta-tests, not functional
tests:

- `tests/test_public_api_freeze.py::test_annotation_facet_reports_hand_written_ndarray_respelling`
- `tests/test_public_api_freeze.py::test_public_api_return_contract` —
  *"Public API contract drift detected in 17 symbol(s)"*
- `tests/test_kinematics_restructure_compat.py::test_moved_method_owners_descriptors_and_signatures_are_preserved`
  — hash mismatch
- `tests/test_kinematics_restructure_compat.py::test_annotation_drift_is_guarded_by_the_ast_pin_not_the_signature_pin`
  — hash mismatch

Root cause of the drift: NumPy 2.5.x changed how `numpy.typing.NDArray` is
stringified on introspection — `numpy.ndarray[tuple[int, ...],
numpy.dtype[numpy.float64]]` (old) vs. bare `NDArray[numpy.float64]` (new).
`tests/test_public_api_freeze.py` has a hardcoded allow-list of known
spellings (around line 481) that doesn't include the new one yet, and the
golden JSON / hashes were generated against the old spelling.

**1109 real functional tests passed** under numba 0.67.0 + numpy 2.5.2 —
CPU-side kinematics/dynamics/backend-dispatch behavior looks unaffected.

## What's still unverified — this is the actual investigation work

- **GPU/CUDA path is untested against this combo.** `cupy-gpu`,
  `torch-cuda`, `jax-cuda`, and `optional-deps` all gate on
  `vars.GPU_RUNNER_ENABLED == 'true'` and a self-hosted runner; they were
  skipped entirely in the run analyzed here. Numba JIT-compiles the CUDA
  kernels — that's exactly the code path most likely to be sensitive to a
  numba minor bump, and it has zero live evidence either way yet.
- **The "17 symbol(s)" drift list was not read in full** — only the two
  visible in the truncated log excerpt were eyeballed. Need to confirm all
  17 are cosmetic annotation-spelling changes and not an actual signature
  change hiding in the same failure.
- Whether NumPy 2.5.x changes anything else relevant (deprecations,
  behavior changes) beyond the typing repr — not checked.

## Why a repo fix alone doesn't help existing users

PyPI package metadata is immutable once published. `ManipulaPy 1.4.0` is
already live with the unbounded `numba>=0.60` pin, so anyone installing it
today already gets numba 0.67.0 + numpy 2.5.2 regardless of what changes
land in this repo. Fixing this for real users requires a **new PyPI
release** (patch: `1.4.1`), not just a merged PR.

Release mechanism in this repo (`.github/workflows/publish.yml`): fires on
a **GitHub Release being published** (not a tag push, not a merge) → builds
sdist/wheel → `gpu-release-gate` (requires live GPU CI evidence) → OIDC
trusted-publish to PyPI. So shipping the fix means: bump version + changelog
→ merge → someone creates and publishes the GitHub Release → gate must pass
on real GPU hardware → PyPI publish is automatic from there.

## Candidate immediate fix (not yet applied)

In `pyproject.toml` (appears twice, lines 71 and 83):

```diff
- "numba>=0.60",
+ "numba>=0.60,<0.67",
```

Plus: bump `version = "1.4.0"` → `"1.4.1"` (line 12), add a
`[Unreleased]` → `[1.4.1]` entry to `CHANGELOG.md` (there is currently no
`[Unreleased]` section — it jumps straight from the top of the file to
`[1.4.0]`).

This has **not** been applied. Before applying, resolve the "still
unverified" items above, since a hard pin below the currently-resolving
version is itself a real user-facing change and deserves the same scrutiny
the drift did.

## Prevention — stop this from happening silently again

The immediate pin is a patch, not a fix for the underlying pattern: any
unbounded transitive-sensitive dependency (numba being the main one, since
it re-pins numpy) can shift ManipulaPy's resolved NumPy version on any day,
with no ManipulaPy commit involved. Options worth evaluating (not decided,
not implemented):

1. **Always keep an upper bound on `numba`**, and bump it deliberately
   (Dependabot/Renovate PR + review) rather than leaving it open.
2. **A scheduled "latest deps" canary CI job** — separate from the pinned
   build that gates PRs/releases — that installs against unconstrained/
   latest transitive deps on a cron and reports drift *before* it silently
   lands in a real user's install. This would have caught the numba 0.67.0
   bump the day it happened instead of via an unrelated PR's CI failing.
3. **Make `test_public_api_freeze.py`'s annotation canonicalization more
   robust** to known-safe NumPy typing-repr changes across versions, so a
   cosmetic stdlib/NumPy rendering change doesn't read as API drift. This
   reduces false-positive noise but does **not** replace the canary job —
   the canary is what surfaces the dependency shift in the first place.
4. Consider whether `numpy>=2.0,<3.0` itself is too wide given how many
   scientific-stack packages (numba included) lag major NumPy releases.

## Relevant references

- PR #83: https://github.com/boelnasr/ManipulaPy/pull/83 (branch
  `claude/code-coverage-alternative-e93cwj`) — where this surfaced, unrelated
  to that PR's actual content (Codecov → coverage-comment-action swap).
- Failing run: `non-tpu-regression (3.12)` / `numpy-cpu` / `torch-cpu` /
  `jax-cpu`, workflow run 31692943775, commit on PR #83.
- Last known-good run (numpy 2.4.6, numba 0.66.0): `main` push run
  31473399544, 2026-08-11T08:29:02Z — all green.
- numba PyPI history: 0.66.0 pins `numpy<2.5,>=1.22`; 0.67.0 (released
  2026-08-11T23:03:08Z) pins `numpy<2.6,>=1.22`.
- numpy 2.5.2 published to PyPI 2026-08-09T13:44:51Z.
- `pyproject.toml` numpy/numba pins: lines 68, 71, 80, 83.
- Golden-signature tests: `tests/test_public_api_freeze.py` (allow-list
  around line 481), `tests/test_kinematics_restructure_compat.py`.
- Release workflow: `.github/workflows/publish.yml`,
  `.github/workflows/gpu-release-gate.yml`.
